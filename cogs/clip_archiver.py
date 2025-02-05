import asyncio
import collections
import discord
import io
import json
import random
import time

from discord import app_commands
from discord.ext import commands
from diffusers.utils import load_image
from helpers import Async_JSON, Checks, CLIP_Archiver, Embeds
from pathlib import Path

class CLIP_Archiver_Cog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.clip_archiver = None
        self.max_download_queue_size = bot.config['max_download_queue_size']
        self.max_local_models = bot.config['max_local_models']
        self.max_queue_size = bot.config['max_queue_size']
        self.user_download_queue = []
        self.user_download_model_names = []
        self.user_image_queue = []

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="bobross", description="Generate an image using a given text prompt.")
    @app_commands.choices(scheduler=[app_commands.Choice(name=scheduler, value=scheduler) for scheduler in ['euler_ancestral', 'dpm_solver_multistep', 
        'dpm_solver_singlestep', 'heun', 'pndm', 'ddpm', 'ddim', 'k_dpm_2', 'k_dpm_2_ancestral', 'dpm_solver_sde', 'unipc_multistep', 'deis_multistep']
    ])
    # Limit users so they dont hog all the resources
    async def generate_image(self, context:commands.Context, prompt:str, preset:str='', height:app_commands.Range[int, 512, 1024]=None, width:app_commands.Range[int, 512, 1024]=None, 
        num_inference_steps:app_commands.Range[int, 1, 30]=None, guidance_scale:float=None, scheduler:app_commands.Choice[str]='', batch_size:app_commands.Range[int, 1, 6]=None, 
        hires_fix:bool=None, hires_strength:app_commands.Range[float, 0.01, 1.0]=None, init_image:str='', init_strength:app_commands.Range[float, 0.01, 1.0]=None, seed:int=None, 
        clip_skip:int=None, lora_and_embeds:str='', model:str='', negative_prompt:str='') -> None:
        
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return
        await context.defer()
        if not self.clip_archiver:
            self.clip_archiver = await CLIP_Archiver.create(
                civitai_token=self.bot.config['civitai_token'],
                default_model=self.bot.config['default_model'],
                models_path=self.bot.config['models_path'],
                default_user_config=self.bot.config['default_user_config'],
                logger=self.bot.logger,
                profiles_path=Path('configs/diffuser_profiles.json'),
                return_images_and_settings=True
            )
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return
        # Set a limit to the image queue
        if self.user_image_queue.count(context.author.id) >= self.max_queue_size:
            embed = Embeds.embed_builder({'title':"Too many requests", 'description':f"You may only have `{self.max_queue_size}` queued generations at a time.", 
                'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return

        self.user_image_queue.append(context.author.id)
        try:
            outputs = await self.clip_archiver(
                prompt=prompt,
                batch_size=batch_size,
                clip_skip=clip_skip,
                guidance_scale=guidance_scale,
                height=height,
                hires_fix=hires_fix,
                hires_strength=hires_strength,
                init_image=init_image,
                init_strength=init_strength,
                lora_and_embeds=lora_and_embeds,
                model=model,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                preset_name=preset,
                scheduler=scheduler,
                user=str(context.author.id),
                seed=seed,
                width=width
            )
            
            # organise outputs
            images = outputs[0]
            settings = outputs[1]
            
            # convert pil.Image to bytes to send over discord without saving to the disk
            image_bytes = []
            for image in images:
                #find a way to put the settings into the image's metadata
                tmp = io.BytesIO()
                image.save(tmp, format='PNG')
                tmp.seek(0)
                image_bytes.append(tmp)
            files = [
                discord.File(fp=image, filename=f"{time.strftime('%Y-%m-%d_%H-%M-%S')}-{image_seed}.png")
                for image, image_seed in zip(image_bytes, settings['seed'])
            ]

            prompt = settings.pop('prompt')
            seeds = settings.pop('seed')

            settings_block = ''
            for item in settings:
                if not isinstance(settings[item], list):
                    settings_block += item + ':' + str(settings[item]) + ' '
                else:
                    settings_block += item + ':' + ' '.join(settings[item]) + ' '

            # respond
            await context.reply(
                f"*/bobross prompt:*`{prompt}` *{settings_block.strip()}* with seeds {seeds} for {context.author.mention}",
                files=files, suppress_embeds=True
            )
        except Exception as exception:
            embed = Embeds.embed_builder({'title':"Error", 'description':exception, 'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
        finally:
            self.user_image_queue.remove(context.author.id)

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="download-model", description="Download a /bobross model from huggingface.")
    async def download_model(self, context:commands.Context, model:str) -> None:
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return
        await context.defer()
        author_id = str(context.author.id)
        if not self.clip_archiver:
            self.clip_archiver = await CLIP_Archiver.create(
                civitai_token=self.bot.config['civitai_token'],
                default_model=self.bot.config['default_model'],
                models_path=self.bot.config['models_path'],
                default_user_config=self.bot.config['default_user_config'],
                logger=self.bot.logger,
                profiles_path=Path('configs/diffuser_profiles.json'),
                return_images_and_settings=True
            )

        # Set a limit to the download queue
        if self.user_download_queue.count(context.author.id) >= self.max_download_queue_size:
            embed = Embeds.embed_builder({'title':"Too many download requests", 'description':f"You may only have `{self.max_download_queue_size}` queued downloads at a time.", 
                'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return

        # Ensure users don't go overboard
        user_models = await self.clip_archiver.model_manager.get_user_models(author_id)
        if self.max_local_models >= 0 and len(user_models) >= self.max_local_models and context.author.id not in self.bot.config['owners']:
            embed = Embeds.embed_builder({'title':f"You have too many models.", 
                'description':f"There may only be {self.max_local_models} models max, remove one from your collection.", 'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return
        
        try:
            download_dict = self.clip_archiver.model_manager.get_download_dict(model)
        except Exception as exception:
            embed = Embeds.embed_builder({'title':"Error retrieving download info:", 'description':exception, 
                'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return
        if download_dict['model_type'] in ['TextualInversion', 'LORA'] and self.clip_archiver.model_manager.models.get(download_dict['model_pipeline']) == None:
            embed = Embeds.embed_builder({'title':f"No Usable Models", 'description':f"There are no `{download_dict['model_pipeline']}` models for you to use `{download_dict['model_type']}` with.", 'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return

        self.user_download_queue.append(context.author.id)
        embed = Embeds.embed_builder({'title':f"Queueing Model Download", 
            'description':f"Downloading of `{download_dict['model_name']}` has been queued at position `{len(self.user_download_queue)}`. Please be patient, this may take a while.", 
            'color':0x9C84EF})
        await context.send(embed=embed, ephemeral=True)
        try:
            model_dict = await self.clip_archiver.model_manager(model, author_id)
            embed = Embeds.embed_builder({'title':None, 'description':f"Successfully downloaded `{model_dict['model_name']}`.", 'color':0x9C84EF})
            await context.send(embed=embed, ephemeral=False)
        except Exception as exception:
            embed = Embeds.embed_builder({'title':"Error downloading model:",'description':exception, 'color':0x9C84EF})
            await context.send(embed=embed, ephemeral=True)
        finally:
            self.user_download_queue.remove(context.author.id)

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="remove-model", description="Remove one of your downloaded /bobross models.")
    async def remove_model(self, context:commands.Context, model:str) -> None:
        await context.defer()
        if not self.clip_archiver:
            self.clip_archiver = await Diffuser.create(
                civitai_token=self.bot.config['civitai_token'],
                default_model=self.bot.config['default_model'],
                models_path=self.bot.config['models_path'],
                default_user_config=self.bot.config['default_user_config'],
                logger=self.bot.logger,
                profiles_path=Path('configs/diffuser_profiles.json'),
                return_images_and_settings=True
            )
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return
            
        model_info = self.clip_archiver.model_manager.get_model_info(model)
        if model_info:
            if model == self.clip_archiver.model_manager.current_download:
                embed = Embeds.embed_builder({'title':f"Model Downloading.", 'description':f"Can't interrupt it, sorry.", 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
                return

            user_models = await self.clip_archiver.model_manager.get_user_models(context)
            if model not in user_models and not context.author.id in self.bot.config['owners']:
                embed = Embeds.embed_builder({'title':f"Model Not Owned.", 'description':f"You can only remove your own models.", 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
                return

            await self.clip_archiver.model_manager.remove_model(model, str(context.author.id))            
            embed = Embeds.embed_builder({'title':f"Model Removed", 'description':f"Successfully removed `{model}` from your collection.", 'color':0x9C84EF})
            await context.send(embed=embed)
        else:
            embed = Embeds.embed_builder({'title':f"Model Not Found", 'description':f"Couldn't remove `{model}`.", 'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="list-models", description="Provides a list of models available to use with /bobross.")
    async def list_models(self, context:commands.Context) -> None:
        await context.defer()
        if not self.clip_archiver:
            self.clip_archiver = await CLIP_Archiver.create(
                civitai_token=self.bot.config['civitai_token'],
                default_model=self.bot.config['default_model'],
                models_path=self.bot.config['models_path'],
                default_user_config=self.bot.config['default_user_config'],
                logger=self.bot.logger,
                profiles_path=Path('configs/diffuser_profiles.json'),
                return_images_and_settings=True
            )
        if not await Checks.channel_allowed(context, self.bot.config['allowed_channels']):
            return
        
        user_models = await self.clip_archiver.model_manager.get_user_models(str(context.author.id))
        newline = '\n'

        models = self.clip_archiver.model_manager.list_models()
        fields = []
        for model_pipeline in models:
            textual_inversion_block = []
            lora_block = []
            model_block = ''

            # go through model types
            for model_type in models[model_pipeline]:
                models_of_type = models[model_pipeline][model_type]
                print(model_type)
                if 'LORA' in model_type: 
                    lora_block.append(', '.join(models_of_type))
                elif 'TextualInversion' in model_type:
                    textual_inversion_block.append(', '.join(models_of_type))
                else:
                    model_block = f"```{', '.join(models_of_type)}```"
            fields.append({'name':f"-- {model_pipeline} --", 'value':model_block, 'inline':False})
            if lora_block:
                lora_block.sort()
                fields.append({'name':"LORA", 'value':f"```{', '.join(lora_block)}```", 'inline':True})
            if textual_inversion_block:
                textual_inversion_block.sort()
                fields.append({'name':"Embeds", 'value':f"```{', '.join(textual_inversion_block)}```", 'inline':True})
        if len(user_models) >= 1 and not context.author.id in self.bot.config['owners']:
            user_models.sort()
            fields.append({'name':"-- Your Model Collection --", 'value':f"```{', '.join(user_models)}```", 'inline':False})
        embed = Embeds.embed_builder({'title':"/bobross models:", 'description':None, 'color':0x9C84EF}, fields)
        await context.send(embed=embed)
        

async def setup(bot):
    await bot.add_cog(CLIP_Archiver_Cog(bot))
