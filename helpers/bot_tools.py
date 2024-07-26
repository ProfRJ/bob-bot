import aiofiles
import asyncio
import discord
import logging
import json
import requests

from discord.ext import commands
from pathlib import Path


class Async_JSON:
    async def async_load_json(pathToLoad) -> dict:
        async with aiofiles.open(Path(pathToLoad), 'r') as file:
            contents = await file.read()
        return json.loads(contents)

    async def async_save_json(pathToSave, data) -> None:
        async with aiofiles.open(Path(pathToSave), 'w') as file:
            await file.write(json.dumps(data, ensure_ascii=False, indent=4))


class Checks:
    def is_owner():
        """Check to see if the user executing the command is an owner of the bot."""
        async def predicate(context: commands.Context) -> bool:
            data = await Async_JSON.async_load_json(Path(__file__).parent.parent / 'configs' / 'config.json')
            if context.author.id not in data['owners']:
                raise Exceptions.UserNotOwner
            return True
        return commands.check(predicate)

    def is_blacklisted():
        """Check to see if the user executing the command is blacklisted."""
        async def predicate(context: commands.Context) -> bool:
            data = await Async_JSON.async_load_json(Path(__file__).parent.parent / 'configs' / 'config.json')
            if context.author.id in data['blacklisted']:
                raise Exceptions.UserBlacklisted
            return True
        return commands.check(predicate)


class Downloads:
    def download_file(file_path, url):
        """Retrive a file from the given url and download it in chunks."""
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        return Path(file_path)


class Embeds:
    def embed_builder(args:dict):
        """Makes a discord embed from given args."""
        embed = discord.Embed(
            title=args['title'],
            description=args['description'], 
            color=args['color']
        )
        return embed


class Exceptions:
    class UserBlacklisted(commands.CheckFailure):
        """Thrown when a user is attempting something, but is blacklisted."""
        def __init__(self, message="User is blacklisted!"):
            self.message = message
            super().__init__(self.message)

    class UserNotOwner(commands.CheckFailure):
        """Thrown when a user is attempting something, but is not an owner of the bot."""
        def __init__(self, message="User is not an owner of the bot!"):
            self.message = message
            super().__init__(self.message)


class Logger:
    # Setup both of the loggers
    def import_logger():
        logger = logging.getLogger("discord_bot")
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(filename="discord.log", encoding="utf-8", mode="w")
        formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s', "%Y-%m-%d %H:%M:%S")
        logger.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger



