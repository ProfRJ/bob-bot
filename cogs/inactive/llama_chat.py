import asyncio
import collections
import discord
import json

from discord import app_commands
from discord.ext import commands
from helpers import Checks, Llama_Chat, Embeds
from pathlib import Path


class Llama_Chat_Cog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.llama_chat_api = None

    async def recursive_reply_search(self, channel_info:dict, message_to_recurse:discord.Message) -> list:
        """
        Searches from the most recent message backwards, collecting the contents of each message into a list to return.
        """    
        reply_chain = collections.deque(maxlen=self.llama_chat_api.max_message_history)
        
        while message_to_recurse.reference:
            message_to_recurse = await message_to_recurse.channel.fetch_message(message_to_recurse.reference.message_id)
            clean_content = message_to_recurse.clean_content
            if message_to_recurse.embeds:
                embed = message_to_recurse.embeds[0]
                fields = []
                if embed.title:
                    fields.append(embed.title)
                if embed.description:
                    fields.append(embed.description)
                if embed.fields:
                    for field in embed.fields:
                        field = f"{field.name} = {field.value}"
                        fields.append(field)
                    clean_content += ', '.join(fields)
            if message_to_recurse.author == self.bot.user:
                reply_chain.appendleft(f"{channel_info['bot_name']}: {clean_content}")
            else:
                user_from_channel = [member for member in message_to_recurse.channel.members if member == message_to_recurse.author]
                if len(user_from_channel) > 0:
                    username = user_from_channel[0].display_name
                else:
                    usrname = message_to_recurse.author.display_name
                reply_chain.appendleft(f"{username}: {clean_content}")
        return list(reply_chain)

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="impersonate", description="Bend the bot to your will by applying a mask of your own making. Enter bot name for default.")
    async def change_bot_identity(self, context:commands.Context, identity:str, description:str='') -> None:
        """ Allows users to change the bot's prompt via command."""
        if not self.llama_chat_api:
            self.llama_chat_api = await Llama_Chat.create(
                bot_prompt=self.bot.config['bot_prompt'],
                llm_model_path=self.bot.config['llm_model_path'],
                n_ctx=self.bot.config['n_ctx'],
                n_gpu_layers=self.bot.config['n_gpu_layers'],
                llm_config=self.bot.config['llm_config'],
                max_message_history=self.bot.config['max_message_history'],
                reply_ratio=self.bot.config['reply_ratio'],
                logger=self.bot.logger
            )
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return

        # Make sure to use the bot's channel specific nickname
        channel_bot_username = [member for member in context.channel.members if member == self.bot.user][0].display_name
        channel_info = self.llama_chat_api.get_channel_info(bot_name=channel_bot_username, channel_id=str(context.channel.id), channel_name=context.channel.name, server_id=str(context.guild.id))
        
        action = None
        if not description:
            if identity == channel_bot_username:
                # Bot name has been used, reverting to its default prompt.
                channel_info['impersonate'] = None
                channel_info['bot_prompt'] = self.bot.config['bot_prompt']
            else:
                # Prepare a request to generate a description with the llm.
                action = 'impersonate'
        else:
            channel_info['impersonate'] = identity
            channel_info['bot_prompt'] = description

        if not action:
            embed = Embeds.embed_builder({'title':f"Welcome, {channel_info['bot_name'] if not channel_info['impersonate'] else channel_info['impersonate']}", 'description':channel_info['bot_prompt'], 'color':0x9C84EF})
        else:
            await context.defer()
            channel_info['impersonate'] = identity
            channel_info['bot_prompt'] = await self.llama_chat_api(content=identity, action=action, channel_info=channel_info)
            embed = Embeds.embed_builder({'title':f"Welcome, {channel_info['bot_name'] if not channel_info['impersonate'] else channel_info['impersonate']}", 'description':channel_info['bot_prompt'], 'color':0x9C84EF})
        await context.reply(embed=embed)

    @commands.Cog.listener('on_message')
    async def listen_on_message(self, message: discord.Message):
        """ Listens for the on_message in bot.py to fire, assigning the Llama_Chat connection if needed and filtering the message. """
        if not self.llama_chat_api:
            self.llama_chat_api = await Llama_Chat.create(
                bot_prompt=self.bot.config['bot_prompt'],
                llm_model_path=self.bot.config['llm_model_path'],
                n_ctx=self.bot.config['n_ctx'],
                n_gpu_layers=self.bot.config['n_gpu_layers'],
                llm_config=self.bot.config['llm_config'],
                max_message_history=self.bot.config['max_message_history'],
                reply_ratio=self.bot.config['reply_ratio'],
                logger=self.bot.logger
            )

        if not await Checks.channel_allowed(message, self.bot.config['allowed_channels'], send_embed=False):
            return
        if message.author == self.bot.user:
            return
        if message.author.id in self.bot.config['blacklisted']:
            return
        
        channel_bot_username = [member for member in message.channel.members if member == self.bot.user][0].display_name
        name_mentioned = any(name in message.clean_content for name in [channel_bot_username, channel_bot_username.title(), 
            channel_bot_username.lower(), channel_bot_username.upper()])

        if (self.bot.user.mentioned_in(message) or name_mentioned) and len(message.clean_content) <= 1800:
            if message.author.bot:
                # stop a possibly infinite conversation between bots
                discardChance = random.randrange(1,5)
                if discardChance == 1:
                    return

            async with message.channel.typing():
                channel_info = self.llama_chat_api.get_channel_info(bot_name=channel_bot_username, channel_id=str(message.channel.id), channel_name=message.channel.name, server_id=str(message.guild.id))
                content = message.clean_content.replace(f'@{channel_bot_username}', channel_bot_username)
                
                if message.reference:
                    reply_list = await self.recursive_reply_search(channel_info=channel_info, message_to_recurse=message)
                else:
                    reply_list = None
                try:
                    response = await self.llama_chat_api(content=content, channel_info=channel_info, reply_list=reply_list, username=message.author.display_name)
                    await message.reply(response)
                except Exception as exception:
                    embed = Embeds.embed_builder({'title':"Error:", 'description':exception, 'color':0xE02B2B})
                    await message.reply(embed=embed)
                
        

async def setup(bot):
    await bot.add_cog(Llama_Chat_Cog(bot))

