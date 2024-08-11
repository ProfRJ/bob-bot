import asyncio
import discord
import json
import os
import sys

from discord.ext import commands
from discord.ext.commands import Bot, Context
from helpers import Checks, Exceptions, Logger
from pathlib import Path


if not (Path(__file__).parent / 'configs' / 'config.json').is_file():
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open(Path(__file__).parent / 'configs' / 'config.json') as file:
        config = json.load(file)  

# Intents
intents = discord.Intents.default()
intents.message_content = True

# Setup the bot
bot = Bot(
    command_prefix=commands.when_mentioned_or(config["command_prefix"]), 
    intents=intents, 
    help_command=None
)
bot.config = config
bot.logger = Logger.import_logger()

@bot.event
async def on_message(message: discord.Message) -> None:
    """
    The code in this event is executed every time someone sends a message, with or without the prefix
    """
    if message.author == bot.user or message.author.bot:
        return
    await bot.process_commands(message)

@bot.event
async def on_command_completion(context: Context) -> None:
    """
    The code in this event is executed every time a normal command has been *successfully* executed.
    """
    full_command_name = context.command.qualified_name
    split = full_command_name.split(" ")
    executed_command = str(split[0])
    bot.logger.info(f"Executed {executed_command} command in {(f'{context.guild.name} (ID: {context.guild.id})' if context.guild else 'DMs')} by {context.author} (ID: {context.author.id})")

@bot.event
async def on_command_error(context: Context, error) -> None:
    """
    The code in this event is executed every time a normal valid command catches an error.
    """
    if isinstance(error, Exceptions.UserBlacklisted):
        ### User Blacklist ###
        embed = Embeds.embed_builder({'title':None, 'description':"You are blacklisted from using the bot!", 'color':0xE02B2B})
        await context.send(embed=embed, ephemeral=True)
        bot.logger.warning(f"{context.author} (ID: {context.author.id}) tried to execute a command in the guild {context.guild.name} (ID: {context.guild.id}), but the user is blacklisted from using the bot.")

    elif isinstance(error, Exceptions.UserNotOwner):
        ### Owner Whitelist ###
        embed = Embeds.embed_builder({'title':None, 'description':"You are not the owner of the bot!", 'color':0xE02B2B})
        await context.send(embed=embed, ephemeral=True)
        bot.logger.warning(f"{context.author} (ID: {context.author.id}) tried to execute an owner only command in the guild {context.guild.name} (ID: {context.guild.id}), but the user is not an owner of the bot.")
    else:
        exception = f"{type(error).__name__}: {error}"
        # Don't show CommandNotFound as this is executed everytime the chatbot is mentioned.
        if not 'CommandNotFound:' in exception:
            bot.logger.error(f"Bot encountered an error, {exception}")

async def load_cogs() -> None:
    cogs_path = Path(__file__).parent / 'cogs'
    cogs = [file.stem for file in cogs_path.iterdir() if file.suffix == '.py']    
    for cog in cogs:
        try:
            await bot.load_extension(f"cogs.{cog}")
            bot.logger.info(f"Loaded extension '{cog}'")
        except Exception as e:
            exception = f"{type(e).__name__}: {e}"
            bot.logger.error(f"Failed to load extension {cog}\n{exception}")

# Wake up the bot
@bot.event
async def on_ready() -> None:
    bot.logger.info(f"Logged in as {bot.user.name}")
    if config["sync_commands_globally"]:
        bot.logger.info("Syncing commands globally...")
        await bot.tree.sync()

if __name__ == '__main__':
    # Run the bot
    asyncio.run(load_cogs())
    bot.run(config["token"])
