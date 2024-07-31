import asyncio
import collections
import discord
import json
import random
from discord import app_commands
from discord.ext import commands
from helpers import Checks, Ctransformer, Embeds
from pathlib import Path

class Ctransformer_Cog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.ctransformer_API = None

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="impersonate", description="Bend the bot to your will by applying a mask of your own making. Enter bot name for default.")
    async def change_bot_identity(self, context:commands.Context, identity:str, description:str='') -> None:
        """ Allows users to change the bot's prompt via slash commands.  """
        if not self.ctransformer_API:
            self.ctransformer_API = await Ctransformer.create(
                bot=self.bot, 
                config=self.bot.config
            )
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return

        action = None
        channel = self.ctransformer_API.get_channel_thread(context.message)
        prompts = channel['prompts']
        default_prompt = self.bot.config['ctransformer_prompt_defaults']['bot_prompt']

        # Make sure to use the bot's channel specific nickname
        channel_bot_user = [member for member in context.message.channel.members if member == self.bot.user][0]

        if not description:
            if identity == channel_bot_user.display_name:
                # Bot name has been used, reverting to its default prompt.
                channel['bot_name'] = identity
                prompts['bot_prompt'] = default_prompt
            else:
                # Prepare an action request to generate a description with the llm.
                action = {'type':"impersonate", 'content':identity}
                message = context.message
        else:
            channel['bot_name'] = identity
            prompts['bot_prompt'] = description

        if not action:
            embed = Embeds.embed_builder({'title':f"Welcome, {channel['bot_name']}", 'description':prompts['bot_prompt'], 'color':0x9C84EF})
            await context.reply(embed=embed)
        else:
            await context.defer()
            await self.ctransformer_API.text_queue.put((context, message, action))

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="die", description="Erases the chat history.")
    async def delete_chat_history(self, context:commands.Context) -> None:
        """Removes the chat history of the origin channel. """
        if not self.ctransformer_API:
            self.ctransformer_API = await Ctransformer.create(
                bot=self.bot, 
                config=self.bot.config
            )
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return
            
        channel = self.ctransformer_API.get_channel_thread(context.message)
        channel['chat_log'] = collections.deque(maxlen=self.bot.config['maxMessageHistory'])
        await context.reply('*falls over*')

    @commands.Cog.listener('on_message')
    async def listen_on_message(self, message: discord.Message):
        """ Listens for the on_message in bot.py to fire, assigning the ctransformer connection if needed and filtering the message. """
        if not self.ctransformer_API:
            self.ctransformer_API = await Ctransformer.create(
                bot=self.bot, 
                config=self.bot.config
            )
        if not await Checks.channel_allowed(message, self.bot.config['allowed_channels'], send_embed=False):
            return
        if message.author == self.bot.user:
            return
        if message.author.id in self.bot.config['blacklisted']:
            return

        name_mentioned = any(name in message.clean_content for name in [self.bot.user.display_name, self.bot.user.display_name.title(), 
            self.bot.user.display_name.lower(), self.bot.user.display_name.upper()])
        if self.bot.user.mentioned_in(message) or name_mentioned and message.clean_content <= 1800:
            context = None
            action = None
            if message.author.bot:
                # stop a possibly infinite conversation between bots
                discardChance = random.randrange(1,10)
                if discardChance == 1:
                    return
            await self.ctransformer_API.text_queue.put((context, message, action))
        

async def setup(bot):
    await bot.add_cog(Ctransformer_Cog(bot))

