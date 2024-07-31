import aiofiles
import asyncio
import collections
import concurrent.futures
import discord
import json
import re
import requests
import time
import torch

from ctransformers import AutoModelForCausalLM
from discord.ext import commands
from helpers import Embeds
from pathlib import Path


class Ctransformer(object):
    @classmethod
    async def create(cls, bot:discord.ext.commands.Bot, config:dict) -> None:
        """
        Creates the worker class, initialising it with the given config.
        """
        self = Ctransformer()
        self.bot = bot
        self.bot.channel_threads = {}
        self.break_idle = asyncio.Event()
        self.config = config
        self.idle = True
        self.idle_manager = asyncio.create_task(self._idle_manager())
        self.text_queue = asyncio.Queue(maxsize=0)
        self.text_worker = asyncio.create_task(self._text_worker())
        return self
        
    def get_channel_thread(self, message:discord.Message) -> dict:
        """
        Retreives data about the channel of the given message, creating an entry if it does not exist.
        """
        author_id = str(message.author.id)
        channel = str(message.channel)
        channel_bot_user = [member for member in message.channel.members if member == self.bot.user][0]
        chat_init = {'chat_log': collections.deque(maxlen=self.bot.config['max_message_history']), 'prompts': self.bot.config['ctransformer_prompt_defaults'].copy(), 'bot_name': channel_bot_user.display_name}

        guild = self.bot.channel_threads.setdefault(message.guild.name, {})
        channel = guild.setdefault(channel, chat_init)
        # If the bot has a nickname in the guild, make sure it is used
        channel['bot_name'] = channel['bot_name'] if not channel['bot_name'] == channel_bot_user.display_name else channel_bot_user.display_name
        channel['channel_bot_user'] = channel_bot_user
        return channel

    async def _idle_manager(self):
        """
        Manages the llm to save memory when not in use.
        """        
        while True:
            self.break_idle.clear()
            if not self.idle:
                # Find a way to free all the GPU memory
                with torch.no_grad():
                    self.bot.logger.info('Waking up the LLM')
                    self.llm = AutoModelForCausalLM.from_pretrained(self.config['ctransformer_model_path'], context_length=self.config['context_length'], 
                        gpu_layers=self.config['gpu_layers'])
                    current_time = time.time()
                    while not current_time-self.last_message_time > 10:
                        await asyncio.sleep(10)
                        current_time = time.time()
                    self.bot.logger.info('Putting the LLM to sleep')
                    self.idle = True
                    del self.llm
                    torch.cuda.empty_cache()
            else:
                await self.break_idle.wait()

    async def _text_worker(self) -> None:
        """
        This asynchronous worker loop is responsible for processing messages one by one from the FIFO text_queue.
        """
        def generate_text(prompt:str) -> str:
            """
            Generate the bot's response based on the prompt.
            """
            response = self.llm(prompt, **self.config['llm_config'])
            return response.strip()

        async def make_prompt(message:discord.Message, channel:dict, action:dict) -> str:
            """
            Combines the channel-wide chat history and recent reply chain with the channel specific prompt for the bot and channel itself.  
            """
            async def recursive_reply_search(message:discord.Message, channel:dict, chat_log:list) -> list:
                """
                Searches from the most recent message backwards, collecting the contents of each message into a list to return.
                """    
                reply_chain = collections.deque(maxlen=self.bot.config['max_message_history'])
                
                while message.reference:
                    message = await message.channel.fetch_message(message.reference.message_id)
                    clean_content = message.clean_content
                    if message.author == self.bot.user:
                        # dodgy code that ensures the bot replies have the right identity in the reply_chain 
                        impersonate = None
                        if clean_content.startswith('(As '):
                            impersonate = clean_content[clean_content.find('(As ')+4:clean_content.find('): ')]
                            clean_content = clean_content[clean_content.find('): ')+3:]
                        if message.embeds:
                            embed = message.embeds[0]
                            clean_content = embed.description
                        reply_chain.appendleft(f"{channel['bot_name'] if not impersonate else impersonate}: {clean_content}")
                    else:
                        reply_chain.appendleft(f"{message.author.display_name}: {message.clean_content}")
                        
                return list(reply_chain)

            chat_log = list(channel['chat_log'])
            chat_log.insert(0, f'--- {message.channel.name} Channel History ---')

            if message.reference:
                reply_chain = await recursive_reply_search(message, channel, chat_log)
                chat_log = [msg for msg in chat_log if msg not in reply_chain]

                # shrink the chat_log in favour of the reply_chain while keeping within the max_message_history
                # works because deques will keep the list within the max_message_history
                if len(reply_chain)-1 + len(chat_log) > self.bot.config['max_message_history']:
                    reply_ratio = int(self.bot.config['reply_ratio']*self.bot.config['max_message_history'])
                    reply_ratio_flipped = int(reply_ratio+self.bot.config['max_message_history']-reply_ratio*2)

                    # if the reply_chain is big enough, crop it to fit within the specified ratio
                    if len(reply_chain) > reply_ratio:
                        excess_reply_chain = len(reply_chain)-reply_ratio 
                        del reply_chain[:excess_reply_chain]

                    # crop the chat_log as the reply_chain grows
                    spare_space = reply_ratio-len(reply_chain)
                    if len(chat_log) > reply_ratio_flipped and spare_space >= 0:
                        del chat_log[:reply_ratio-spare_space]
                    if len(chat_log) > reply_ratio_flipped and spare_space < 0:
                        del chat_log[:reply_ratio]
                                 
                reply_chain.insert(0, "--- Recent Reply Chain History ---")
                chat_log.extend(reply_chain)

            prompts = channel['prompts'] # Keeping prompts as a dict for redundancy even if it holds one value atm
            bot_prompt = f"A few things to keep in mind about {channel['bot_name']}: {prompts['bot_prompt']}"
            
            # If the message is from a command, use the right prompt for it, otherwise use the default chatbot prompt.
            if action == None: 
                clean_content = message.clean_content.replace(channel['channel_bot_user'].display_name, channel['bot_name'])
                chat_log.append(f"{message.author.display_name}: {clean_content}")
                response_string = f"Answer {message.author.display_name}'s latest message like {channel['bot_name']} would. SHORT casual responses are preferable"
                response_prompt = bot_prompt+" "+response_string
                chat_log.insert(0, response_prompt)
                chat_log.append(f"{channel['bot_name']}:")
            else:
                if action['type'] == 'impersonate':
                    clean_content = message.clean_content.replace(channel['channel_bot_user'].display_name, channel['bot_name'])
                    chat_log.append("--- System Character Creator ---")
                    chat_log.append(f"Given the name {action['content']}, write a short description about {action['content']} that includes striking elements for character and background about {action['content']}.")
                    chat_log.append(f"{action['content']}:")

            prompt = '\n'.join(chat_log)
            return prompt, clean_content

        async def process_message(context:discord.ext.commands.Context, message:discord.Message, action:dict) -> None:
            """
            Prepare and send a response for the user message using a channel specific prompt.
            """
            async with message.channel.typing():
                # prepare response
                channel = self.get_channel_thread(message)
                prompt, clean_content = await make_prompt(message, channel, action)
                response = ""
                while response == "":
                    response = await asyncio.to_thread(generate_text, prompt)

                #respond
                if action == None:
                    await message.reply(f"(As {channel['bot_name']}): {response}" if not channel['bot_name'] == channel['channel_bot_user'].display_name else response)
                    # save the message pair together for better consistency
                    channel['chat_log'].append(f'{message.author.display_name}: {clean_content}')
                    channel['chat_log'].append(f"{channel['bot_name']}: {response}")
                    self.bot.logger.info(f"Responded to message in {(f'{message.guild.name} (ID: {message.guild.id})')} by {message.author} (ID: {message.author.id})")
                else:
                    if action['type'] == 'impersonate':
                        prompts = channel['prompts']
                        channel['bot_name'] = action['content']
                        prompts['bot_prompt'] = response
                        embed = Embeds.embed_builder({'title':f"Welcome, {channel['bot_name']}", 'description':response, 'color':0x9C84EF})
                        await context.reply(embed=embed)

        while True:
            ### Worker Loop ###
            try:
                context, message, action = await self.text_queue.get()
                self.last_message_time = time.time()
                if self.idle:
                    self.idle = False
                    self.break_idle.set()
                await process_message(context, message, action)
            except Exception as e:
                exception = f"{type(e).__name__}: {e}"
                self.bot.logger.error(f"Ctransformer encountered an error while replying \n{exception}")
