import asyncio
import discord

from discord import app_commands
from discord.ext import commands
from discord.ext.commands import Bot, Context
from helpers import Checks


class Management(commands.Cog, name="management"):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="ping")
    @Checks.is_owner()
    async def ping(self, context: Context) -> None:
        embed = discord.Embed(
            title="🏓 Pong!",
            description=f"The bot latency is {round(self.bot.latency * 1000)}ms.",
            color=0x9C84EF,
        )
        await context.send(embed=embed)

    @commands.command(name="cog")
    @Checks.is_owner()
    async def manipulate_cogs(self, context: Context, action: str, cog: str) -> None:
        ### manage cogs ###
        if action == 'load':
            try:
                await self.bot.load_extension(f"cogs.{cog}")
            except Exception:
                embed = discord.Embed(description=f"Could not Load the `{cog}` cog.", color=0xE02B2B)
                await context.send(embed=embed)
                return
        if action == 'unload':
            try:
                await self.bot.unload_extension(f"cogs.{cog}")
            except Exception:
                embed = discord.Embed(description=f"Could not {action} the `{cog}` cog.", color=0xE02B2B)
                await context.send(embed=embed)
                return
        if action == 'reload':
            try:
                await self.bot.reload_extension(f"cogs.{cog}")
            except Exception:
                embed = discord.Embed(description=f"Could not {action} the `{cog}` cog.", color=0xE02B2B)
                await context.send(embed=embed)
                return
        embed = discord.Embed(description=f"Successfully {action}ed the `{cog}` cog.", color=0x9C84EF)
        await context.send(embed=embed)
    
    @commands.command(name='slash')
    @Checks.is_owner()
    async def slash_synchronisation(self, context: Context, scope: str, action: str) -> None:
        # synchonizes the slash commands
        async def send_error_message(context, scope):
            embed = discord.Embed(description=
                "The scope must be `global` or `guild`." if ('global' or 'guild') not in scope else
                "The action must be `sync` or `unsync`.", 
                color=0xE02B2B)
            await context.send(embed=embed)

        async def send_success_message(context, scope, action):
            embed = discord.Embed(description=f"{scope} commands have been {action}ed.", color=0x9C84EF)
            await context.send(embed=embed)

        if scope == 'global':
            if action == 'sync':
                await context.bot.tree.sync()
            elif action == 'unsync':
                context.bot.tree.clear_commands(guild=None)
                await context.bot.tree.sync()
            else:
                await send_error_message(context, scope)
                return
            
        elif scope == 'guild':
            if action == 'sync':
                context.bot.tree.copy_global_to(guild=context.guild)
                await context.bot.tree.sync(guild=context.guild)
            elif action == 'unsync':
                context.bot.tree.clear_commands(guild=context.guild)
                await context.bot.tree.sync(guild=context.guild)
            else:
                await send_error_message(context, scope)
                return
        else:
            await send_error_message(context, scope)
            return

        await send_success_message(context, scope, action)


async def setup(bot):
    await bot.add_cog(Management(bot))