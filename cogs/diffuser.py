import asyncio
import collections
import discord
import json
import random
import time

from discord import app_commands
from discord.ext import commands
from helpers import Async_JSON, Checks, Diffuser, Embeds
from pathlib import Path

class Diffuser_Cog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.diffuser_API = None
 
    @Checks.is_blacklisted()
    @commands.hybrid_command(name="bobross", description="Generate an image using a given text prompt.")
    @app_commands.choices(scheduler=[app_commands.Choice(name=scheduler, value=scheduler) for scheduler in ['euler_ancestral', 'dpm_solver_multistep', 
        'dpm_solver_singlestep', 'heun', 'pndm', 'ddpm', 'ddim', 'k_dpm_2', 'k_dpm_2_ancestral', 'dpm_solver_sde', 'unipc_multistep', 'deis_multistep']
    ])
    # Limit users so they dont hog all the resources
    async def generate_image(self, context:commands.Context, prompt:str, preset:str='', height:app_commands.Range[int, 512, 1024]=None, width:app_commands.Range[int, 512, 1024]=None, 
        num_inference_steps:app_commands.Range[int, 1, 30]=None, guidance_scale:float=None, scheduler:app_commands.Choice[str]='', batch_size:app_commands.Range[int, 1, 6]=None, 
        seed:int=None, clip_skip:int=None, lora_and_embeds:str='', model:str='', negative_prompt:str='') -> None:
        if not self.diffuser_API:
            self.diffuser_API = await Diffuser.create(
                bot=self.bot,
                config=self.bot.config
            )

        # make sure there are usable models
        if len(self.diffuser_API.model_manager.models) == 0:
            embed = Embeds.embed_builder({'title':"Hold on...", 'description':"There are no models yet! Please wait while the default model downloads.", 'color':0xE02B2B})
            await context.send(embed=embed)
            return
        # Set a limit to the queue
        if self.diffuser_API.user_queue.count(context.author.id) >= self.bot.config['max_queue_size']:
            embed = Embeds.embed_builder({'title':"Too many requests", 'description':f"You may only have {self.bot.config['max_queue_size']} queued requests at a time.", 
                'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return

        # Correct args
        if clip_skip == 0:
            clip_skip = str(clip_skip)

        # Ensure the user's profile exists
        user_profile = self.diffuser_API.user_profiles.setdefault(str(context.author.id), {'intermediate': self.bot.config['diffuser_default_user_config']})

        # Retrieve the preset from the user's profile or use the default
        preset_name = preset if preset else 'intermediate'
        preset = user_profile.get(preset_name, user_profile['intermediate'])
        preset['model'] = preset.setdefault('model', self.diffuser_API.model_manager.default_model)
        model_info = self.diffuser_API.model_manager.get_model_info(model)
        if not model_info:
            model = preset['model']
            model_info = self.diffuser_API.model_manager.get_model_info(model)
        
        settings = {
        'height':round(height/64)*64 if height else preset['height'],
        'width':round(width/64)*64 if width else preset['width'],
        'num_inference_steps':num_inference_steps if num_inference_steps else preset['num_inference_steps'],
        'guidance_scale':guidance_scale if guidance_scale else preset['guidance_scale'],
        'scheduler':scheduler.value if scheduler else preset['scheduler'],
        'batch_size':batch_size if batch_size else preset['batch_size'],
        'negative_prompt':negative_prompt if negative_prompt else preset['negative_prompt'],
        'seed':seed if seed else preset['seed'],
        'clip_skip':clip_skip if clip_skip else preset['clip_skip'],
        'lora_and_embeds':[lora_or_embed for lora_or_embed in lora_and_embeds.split(' ') if self.diffuser_API.model_manager.get_model_info(lora_or_embed.split(':')[0])] if lora_and_embeds else preset['lora_and_embeds'],
        'model':model
        }

        # Correct args
        if clip_skip == '0':
            settings['clip_skip'] = None
        if settings['seed']:
            if settings['seed'] <= -1:
                settings['seed'] = None

        # Update the user profile with the new settings
        user_profile[preset_name] = settings
        if not preset_name == 'intermediate':
            user_profile['intermediate'] = settings
        await Async_JSON.async_save_json(self.diffuser_API.profiles_path, self.diffuser_API.user_profiles)

        # Get model specific settings
        settings_to_pipe = {
        'prompt':prompt,
        'height':settings['height'],
        'width':settings['width'],
        'num_inference_steps':settings['num_inference_steps'],
        'guidance_scale':settings['guidance_scale'],
        'batch_size':settings['batch_size'],
        'negative_prompt':settings['negative_prompt'],
        'model': settings['model']
        }
        if not settings['seed']:
            settings_to_pipe['seed'] = int(time.time())
        else:
            settings_to_pipe['seed'] = settings['seed']
        if 'SD' in model_info['model_pipeline']:
            settings_to_pipe.update({'scheduler':settings['scheduler'], 'clip_skip':settings['clip_skip'], 'lora_and_embeds':settings['lora_and_embeds']})

        await context.defer()
        self.diffuser_API.user_queue.append(context.author.id)
        await self.diffuser_API.image_queue.put((context, settings_to_pipe))

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="download-model", description="Download a /bobross model from huggingface.")
    async def download_model(self, context:commands.Context, model:str, model_version:int=1) -> None:
        if not self.diffuser_API:
            self.diffuser_API = await Diffuser.create(
                bot=self.bot,
                config=self.bot.config
            )

        # Set a limit to the download queue
        if self.diffuser_API.model_manager.user_download_queue.count(context.author.id) >= self.bot.config['max_download_queue_size']:
            embed = Embeds.embed_builder({'title':"Too many download requests", 'description':f"You may only have `{self.bot.config['max_download_queue_size']}` queued downloads at a time.", 
                'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return
        if model in self.diffuser_API.model_manager.user_download_queue_models:
            embed = Embeds.embed_builder({'title':f"This model is already queued for download.", 'description':f'Please wait until it finishes.', 'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return

        user_models = await self.diffuser_API.model_manager.get_user_models(context)
        if self.bot.config['max_diffuser_models'] >= 0 and len(user_models) >= self.bot.config['max_diffuser_models'] and context.author.id not in self.bot.config['owners']:
            embed = Embeds.embed_builder({'title':f"You have too many models.", 
                'description':f"There may only be {self.bot.config['max_diffuser_models']} models max, remove one from your collection or organise a riot.", 'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)
            return

        download_dict = self.diffuser_API.model_manager.get_download_dict(model, model_version)
        if download_dict:
            if download_dict['model_name'] in user_models:
                embed = Embeds.embed_builder({'title':f"You already own {download_dict['model_name']}.", 'description':None, 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
                return
            if not "SD" in download_dict['model_pipeline']:
                embed = Embeds.embed_builder({'title':f"Unsupported Model", 'description':f"The bot only accepts stable diffusion models.", 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
                return
            if download_dict['model_type'] in ['TextualInversion', 'LORA'] and self.diffuser_API.model_manager.models.get(download_dict['model_pipeline']) == None:
                embed = Embeds.embed_builder({'title':f"No Usable Models", 'description':f"There are no `{download_dict['model_pipeline']}` models for you to use `{download_dict['model_type']}` with.", 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
                return
            if self.diffuser_API.model_manager.get_model_info(download_dict['model_name']):
                model_pipeline = self.diffuser_API.model_manager.models.get(download_dict['model_pipeline'])
                model = model_pipeline.get(download_dict['model_name'])
                model['users'].append(context.author.id)
                await Async_JSON.async_save_json(self.diffuser_models_path / 'diffuser_models.json', self.diffuser_API.models)
                embed = Embeds.embed_builder({'title':f"Model Bumped", 'description':f"It already exists, but thanks to your efforts it will be kept available.", 
                    'color':0x9C84EF})
                await context.send(embed=embed, ephemeral=True)
                return

            await context.defer()
            self.diffuser_API.model_manager.user_download_queue.append(context.author.id)
            self.diffuser_API.model_manager.user_download_queue_models.append(download_dict['model_name'])
            await self.diffuser_API.model_manager.model_queue.put((context, download_dict))
            if self.diffuser_API.model_manager.current_download:
                embed = Embeds.embed_builder({'title':f"Queueing Model Download", 
                    'description':f"Downloading of `{model}` has been queued at position `{len(self.diffuser_API.model_manager.user_download_queue)}`. Please be patient, this may take a while.", 
                    'color':0x9C84EF})
                await context.send(embed=embed, ephemeral=True)
        else:
            embed = Embeds.embed_builder({'title':f"Model Not Found", 
                'description':f"Only civitai urls are accepted with the exception of repo ids of the base stable diffusion models (1.5 to SDXL) from huggingface.", 
                'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="remove-model", description="Remove one of your downloaded /bobross models.")
    async def remove_model(self, context:commands.Context, model:str) -> None:
        if not self.diffuser_API:
            self.diffuser_API = await Diffuser.create(
                bot=self.bot,
                config=self.bot.config
            )
        model_info = self.diffuser_API.model_manager.get_model_info(model)
        if model_info:
            if model == self.diffuser_API.model_manager.current_download:
                embed = Embeds.embed_builder({'title':f"Model Downloading.", 'description':f"Can't interrupt it, sorry.", 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
                return

            user_models = await self.diffuser_API.model_manager.get_user_models(context)
            if model not in user_models and not context.author.id in self.bot.config['owners']:
                embed = Embeds.embed_builder({'title':f"Model Not Owned.", 'description':f"You can only remove your own models.", 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
                return

            await self.diffuser_API.model_manager.remove_model(model)            
            embed = Embeds.embed_builder({'title':f"Model Removed", 'description':f"Successfully removed `{model}` from your collection.", 'color':0x9C84EF})
            await context.send(embed=embed)
        else:
            embed = Embeds.embed_builder({'title':f"Model Not Found", 'description':f"Couldn't remove `{model}`.", 'color':0xE02B2B})
            await context.send(embed=embed, ephemeral=True)

    @Checks.is_blacklisted()
    @commands.hybrid_command(name="list-models", description="Provides a list of models available to use with /bobross.")
    async def list_models(self, context:commands.Context) -> None:
        if not self.diffuser_API:
            self.diffuser_API = await Diffuser.create(
                bot=self.bot,
                config=self.bot.config
            )
        user_models = await self.diffuser_API.model_manager.get_user_models(context)
        newline = '\n'

        models = []
        model_pipelines = [pipeline for pipeline in self.diffuser_API.model_manager.models]
        model_pipelines.sort()
        # go through the model versions
        for model_pipeline in model_pipelines:
            model_types = [model_type for model_type in self.diffuser_API.model_manager.models.get(model_pipeline)]
            model_types.sort()
            model_block = ''
            lora_and_embed_block = []
            for model_type in model_types: 
                models_of_type = self.diffuser_API.model_manager.models.get(model_pipeline).get(model_type)
                if 'Checkpoint' not in model_type: 
                    lora_and_embed_block.append(', '.join(models_of_type))
                else:
                    model_block = f"**{model_pipeline}:** ```{', '.join(models_of_type)}```"
            models.append(model_block)
            if lora_and_embed_block:
                lora_and_embed_block.sort()
                models.append(f"*LORAs and Embeds* ```{', '.join(lora_and_embed_block)}```")
        if len(user_models) >= 1 and not context.author.id in self.bot.config['owners']:
            user_models.sort()
            models.append(f"**Your Model Collection:**```{', '.join(user_models)}```")
        embed = Embeds.embed_builder({'title':"/bobross models:", 'description':f"{newline.join(models)}", 'color':0x9C84EF})
        await context.send(embed=embed)
        

async def setup(bot):
    await bot.add_cog(Diffuser_Cog(bot))
