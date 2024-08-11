import asyncio
import concurrent.futures
import discord
import io
import json
import shutil
import time
import torch
import requests

from compel import Compel, DiffusersTextualInversionManager
from datetime import datetime
# import the diffuser pipelines
from diffusers import (AutoencoderKL, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusion3Pipeline)
# import the diffuser schedulers
from diffusers import (EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, HeunDiscreteScheduler, PNDMScheduler,
    DDPMScheduler, DDIMScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler, DEISMultistepScheduler)
from diffusers.utils import load_image
from discord.ext import commands
from helpers import Async_JSON, Downloads, Embeds, vae_pt_to_vae_diffuser
from huggingface_hub import model_info, try_to_load_from_cache, _CACHED_NO_EXIST
from pathlib import Path
from PIL import Image


class Model_Manager():
    @classmethod
    async def create(cls, civitai_token='', default_model:str='runwayml/stable-diffusion-v1-5', diffuser_models_path:str='', logger=None, 
        owners:[int,str]=['system']) -> None:
        """
        Creates and returns an asyncio model_manager object.
        diffuser_models_path - str or pathlike object with the intended location for the models (default '')
        default_model - huggingface repo id or a link to a model hosted on civit (default 'runwayml/stable-diffusion-v1-5')
        logger - logging object (default None)
        """
        self = Model_Manager()
        self.civitai_token = civitai_token
        self.current_download = None
        self.default_model = default_model
        self.diffuser_models_path = Path(diffuser_models_path) if not diffuser_models_path == '' else Path(__file__).parent.parent / 'models'
        self.model_queue = asyncio.Queue(maxsize=0)
        self.download_worker = asyncio.create_task(self._download_worker())
        self.logger = logger
        self.models = await self._get_models()
        self.owners = owners
        self.user_download_queue = []
        self.user_download_queue_models = []
        return self

    async def _get_models(self) -> dict:
        """ 
        Initialise the model manager with the last saved list of models, and will add a download if it is empty.
        """
        # Make sure the paths exist.
        if not self.diffuser_models_path.is_dir():
            self.diffuser_models_path.mkdir(parents=True, exist_ok=True)
        if not (self.diffuser_models_path / 'diffuser_models.json').is_file():
            with open(self.diffuser_models_path / 'diffuser_models.json', 'w') as f:
                json.dump({}, f)
        # Get models downloaded from previous sessions and see if the default model is among them.
        cached_models = await Async_JSON.async_load_json(self.diffuser_models_path / 'diffuser_models.json')
        default_model_dict = self.get_download_dict(self.default_model)
        self.default_model = default_model_dict['model_name']
        model_names = []
        for model_pipeline in cached_models:
            model_types = cached_models.get(model_pipeline)
            for model in model_types['Checkpoint']:
                model_names.append(model)
        # If it isn't, add it to the download_queue and return the cache_model dict.
        if not default_model_dict['model_name'] in model_names:
            await self.model_queue.put((None, default_model_dict))
        return cached_models

    def get_download_dict(self, url_or_repo:str, model_version:int=1):
        def correct_hf_name(repo_id:str) -> str:
            """Format the name of the hf repo."""
            repo_name = repo_id.split('/')[-1]
            repo_name_list = repo_name.split('-')
            model_name = ""
            for block in repo_name_list:
                index = repo_name_list.index(block)
                if not index == 0:
                    last_char = repo_name_list[index-1][-1]
                    seperator = '.' if last_char.isnumeric() else ' '
                    block = seperator + block
                model_name = model_name + block
            model_name = model_name.title()
            return model_name.replace(' ', '')

        download_dict = {}
        # If it's not a civitai address, assume it's one of the allowed huggingface repos.
        if not 'civitai.com/models' in url_or_repo: 
            model_name = correct_hf_name(url_or_repo)
            hf_info = model_info(url_or_repo)
            if 'stable-diffusion-v1-5' in url_or_repo:
                model_pipeline = 'SD 1'
            if 'stable-diffusion-2' in url_or_repo:
                model_pipeline = 'SD 2'
            if 'stable-diffusion-xl-base' in url_or_repo:
                model_pipeline = 'SDXL'
            try:
                download_dict.update({'model_pipeline':model_pipeline, 'model_type':"Checkpoint", 'model_name':model_name, 'url_or_repo':url_or_repo, 'vae_dict':None})
            except:
                return None
        else:
            model_id = url_or_repo.split('/')[-1]
            api_url = f'https://civitai.com/api/v1/models/{model_id}'
            response = requests.get(api_url)
            response = response.json()

            if model_version <= 0:
                model_version = 1
            model_version -= 1

            # civitai model versions are sorted latest to oldest
            try:
                chosen_model = response['modelVersions'][model_version]
            except:
                chosen_model = response['modelVersions'][-1]
            chosen_file = [file for file in chosen_model['files'] if file['type'] == 'Model'][0]
            vae_list = [file for file in chosen_model['files'] if file['type'] == 'VAE']
            if 'SD' in chosen_model['baseModel']: 
                if response['type'] in ['Checkpoint', 'TextualInversion', 'LORA']:
                    file_name = chosen_file['name']
                    url_or_repo = chosen_model['downloadUrl'] + f'?token={self.civitai_token}'
                    download_path = self.diffuser_models_path / model_pipeline / file_name if 'Checkpoint' in response['type'] else self.diffuser_models_path / model_pipeline / response['type'] / file_name
                        
                    if 'SD 1' in chosen_model['baseModel']:
                        model_pipeline = 'SD 1'
                    elif 'SD 2' in chosen_model['baseModel']:
                        model_pipeline = 'SD 2'
                    elif 'SDXL' in chosen_model['baseModel']:
                        model_pipeline = 'SDXL'

                    # include the vae if a model is listed with it
                    if vae_list:
                        latest_vae = vae_list[0]
                        vae_name = latest_vae['name']
                        vae_url = latest_vae['downloadUrl']
                        vae_download_path = self.diffuser_models_path / vae_name
                        vae_dict = {'url':vae_url, 'download_path':str(vae_download_path), 'base_model':chosen_model['baseModel']}

                    download_dict.update({'model_pipeline':model_pipeline, 'model_type':response['type'], 'model_name':file_name, 'url_or_repo':url_or_repo, 
                        'download_path':str(download_path), 'vae_dict':vae_dict if vae_list else None, 
                        'embedding_trigger':chosen_model['trainedWords'] if response['type'] == 'TextualInversion' else None})

        download_dict['model_name'] = download_dict['model_name'].replace('.safetensors','').replace('.ckpt','').replace('.pt','').replace('-','_')
        return download_dict

    def get_model_info(self, model:str) -> dict:
        """
        Search existing models to see if exists, and return it's settings. 
        """
        model_info = None
        for model_pipeline in self.models:
            model_types = self.models.get(model_pipeline)
            for model_type in model_types:
                models_of_type = self.models.get(model_pipeline).get(model_type)
                for stored_model in models_of_type:
                    if model in models_of_type:
                        model_dict = models_of_type.get(stored_model)
                        model_info = {'model_pipeline': model_pipeline, 'model_type':model_type, 'path': model_dict['path'], 'users': model_dict['users']}
                        if 'TextualInversion' in model_type:
                            model_info.update({'embedding_trigger': model_dict['embedding_trigger']})
        return model_info

    def get_pipeline(self, pipeline:str):
        """ 
        Retrieve the specific pipeline for the model.
        """
        pipelines = {
        "SD 1": StableDiffusionPipeline,
        "SD 1Img2Img": StableDiffusionImg2ImgPipeline,
        "SD 2": StableDiffusionPipeline,
        "SD 2Img2Img": StableDiffusionImg2ImgPipeline,
        "SDXL": StableDiffusionXLPipeline,
        "SDXLImg2Img": StableDiffusionXLImg2ImgPipeline
        }
        return pipelines.get(pipeline)

    def get_scheduler(self, scheduler:str):
        """
        Retrieve the specified scheduler.
        """
        schedulers = {
        "euler_ancestral": EulerAncestralDiscreteScheduler,
        "dpm_solver_multistep": DPMSolverMultistepScheduler,
        "dpm_solver_singlestep": DPMSolverSinglestepScheduler,
        "heun": HeunDiscreteScheduler,
        "pndm": PNDMScheduler,
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "k_dpm_2": KDPM2DiscreteScheduler,
        "k_dpm_2_ancestral": KDPM2AncestralDiscreteScheduler,
        "dpm_solver_sde": DPMSolverSDEScheduler,
        "unipc_multistep": UniPCMultistepScheduler,
        "deis_multistep": DEISMultistepScheduler
        }
        return schedulers.get(scheduler)

    async def get_user_models(self, context:discord.ext.commands.Context) -> list:
        """ 
        Retrieve any models a user may have downloaded. 
        """
        user_models = []
        for model_pipeline in self.models:
            model_types = self.models.get(model_pipeline)
            for model_type in model_types:
                models_of_type = model_types.get(model_type)
                for model in models_of_type:
                    if model in models_of_type:
                        model_dict = models_of_type.get(model)
                        if not context.author.id in model_dict['users']:
                            pass
                        else:
                            user_models.append(model)
        return user_models

    async def remove_model(self, model, context=None):
        model_info = self.get_model_info(model) # refactor to accept the dict
        model_pipeline = self.models.get(model_info['model_pipeline'])
        model_type = model_pipeline.get(model_info['model_type'])
        model_entry = model_type.get(model)
        if context:
            model_entry['users'].remove(context.author.id)
        else:
            model_entry['users'] = []

        # Remove the model entry and delete it if no one else is using it.
        if len(model_entry['users']) == 0:
            model_type.pop(model)
            if '/snapshots/' in model_info['path']:
                model_path = Path(model_info['path']).parent.parent
            else: 
                model_path = Path(model_info['path'])

            if 'Checkpoint' in model_info['model_type']: 
                shutil.rmtree(model_path)
            else:
                model_path.unlink()

            if len(model_type) <= 0:
                model_pipeline.pop(model_info['model_type'])
                if len(model_pipeline) <= 0:
                    self.models.pop(model_info['model_pipeline'])
        await Async_JSON.async_save_json(self.diffuser_models_path / 'diffuser_models.json', self.models)

    async def _download_worker(self) -> None:
        """
        Asyncio download worker that retrieves items from the download queue.
        """
        def download_model(download_dict:dict) -> dict:
            """
            Uses a diffuers pipeline to manage the download if its from huggingface, else download it from civitai and turn that checkpoint file into the diffusers format.
            """
            self.current_download = download_dict['model_name']

            # Use the diffuser pipeline to manage the model download of a huggingface diffuser model.
            if not 'civitai.com' in download_dict['url_or_repo']:
                pipeline = self.get_pipeline(download_dict['model_pipeline'])
                local_model_pipeline = pipeline.from_pretrained(
                    download_dict['url_or_repo'],
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                    cache_dir=self.diffuser_models_path / download_dict['model_pipeline']
                )

                model_index_path = try_to_load_from_cache(download_dict['url_or_repo'], filename='model_index.json', cache_dir=self.diffuser_models_path)
                diffuser_path = Path(model_index_path).parent
                model_dict = {'model_pipeline':download_dict['model_pipeline'], 'model_type':download_dict['model_type'], 'model_name':download_dict['model_name'], 'path':str(diffuser_path)}
            # Download the model from civitai and turn it into a diffuser model.
            else:
                # Get our files downloaded.
                (self.diffuser_models_path / download_dict['model_pipeline']).mkdir(parents=True, exist_ok=True)
                if not (self.diffuser_models_path / download_dict['model_pipeline'] / download_dict['model_type']).is_dir() and 'Checkpoint' not in download_dict['model_type']:
                    (self.diffuser_models_path / download_dict['model_type']).mkdir(parents=True, exist_ok=True)
                model_path = Downloads.download_file(download_dict['download_path'], download_dict['url_or_repo'])

                if 'Checkpoint' in download_dict['model_type']:
                    if download_dict['vae_dict']:
                        vae_dict = download_dict.get('vae_dict')
                        vae_checkpoint_path = Downloads.download_file(vae_dict['download_path'], vae_dict['url'])
                    
                    pipeline = self.get_pipeline(download_dict['model_pipeline'])
                    # Try to use fp16 weights
                    try:
                        local_model_pipeline = pipeline.from_single_file(
                            model_path,
                            torch_dtype=torch.float16,
                            variant="fp16",
                            use_safetensors=True,
                            cache_dir=self.diffuser_models_path / download_dict['model_pipeline']
                        )
                    except:
                        local_model_pipeline = pipeline.from_single_file(
                            model_path,
                            torch_dtype=torch.float16,
                            use_safetensors=True,
                            cache_dir=self.diffuser_models_path / download_dict['model_pipeline']
                        )
                    model_path.unlink()
                    
                    local_model_pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
                    model_path = str(self.diffuser_models_path / download_dict['model_name'].lower())
                    local_model_pipeline.save_pretrained(save_directory=model_path)

                    if download_dict['vae_dict']:
                        vae_subfolder = model_path+'/vae'
                        vae_dict.update({'save_pretrained_vae_path':vae_subfolder})
                        vae_pt_to_vae_diffuser(vae_dict)
                        vae_checkpoint_path.unlink()
                    del local_model_pipeline

                model_dict = {'model_pipeline':download_dict['model_pipeline'], 'model_type':download_dict['model_type'], 'model_name':download_dict['model_name'], 
                    'path':str(model_path)}     
                if 'TextualInversion' in download_dict['model_type']:
                   model_dict.update({'embedding_trigger': download_dict['embedding_trigger'][0]})

            torch.cuda.empty_cache()
            self.current_download = None
            return model_dict
            
        async def process_download(context:discord.ext.commands.Context, download_dict:dict):
            if context:
                embed = Embeds.embed_builder({'title':f"Model Download Started", 'description':f"Downloading `{download_dict['model_name']}`.", 
                    'color':0x9C84EF})
                await context.reply(embed=embed, ephemeral=True)
            # Add the entry to the class' dict
            model_dict = await asyncio.to_thread(download_model, download_dict)
            model_pipeline = self.models.setdefault(model_dict['model_pipeline'], {})
            model_type = model_pipeline.setdefault(model_dict['model_type'], {})
            model_entry = model_type.setdefault(model_dict['model_name'], {"path":model_dict['path'], "users":[]})
            if 'TextualInversion' in download_dict['model_type']:
                model_entry.update({'embedding_trigger':model_dict['embedding_trigger']})

            # If there is context, a user has performed this action so let them know it is done and assign the model to them.
            if context:
                model_entry['users'].append(context.author.id)
                embed = Embeds.embed_builder({'title':f"{model_dict['model_type']} Downloaded", 'description':f"Successfully downloaded `{model_dict['model_name']}` for {context.author.mention}.", 
                    'color':0x9C84EF})
                await context.reply(embed=embed)
            else:
                model_entry['users'].extend(self.owners)

            await Async_JSON.async_save_json(self.diffuser_models_path / 'diffuser_models.json', self.models)
            if self.logger:
                    self.logger.info(f"{model_dict['model_name']} ({model_dict['path']}) added.")

        while True:
            try:
                context, download_dict = await self.model_queue.get()
                await process_download(context, download_dict)    
            except Exception as exception:
                if context:
                    embed = Embeds.embed_builder({'title':f"Failed to Download Model", 'description':f"Failed to download `{download_dict['model_name']}` o7", 
                        'color':0xE02B2B})
                    await context.reply(embed=embed, ephemeral=True)
                if self.logger:
                    self.logger.error(f"Model manager encountered an error while adding {download_dict['model_name']}:\n{exception}")
            if context:
                self.user_download_queue.remove(context.author.id)
                self.user_download_queue_models.remove(download_dict['model_name'])


def resize_and_crop_centre(images:[Image], new_width:int, new_height:int) -> Image:
    """Rescales an image to different dimensions without distorting the image."""
    if not isinstance(images, list):
        images = [images]

    rescaled_images = []
    for image in images:
        img_width, img_height = image.size
        width_factor, height_factor = new_width/img_width, new_height/img_height
        factor = max(width_factor, height_factor)
        image = image.resize((int(factor*img_width), int(factor*img_height)), Image.LANCZOS)

        img_width, img_height = image.size
        width_factor, height_factor = new_width/img_width, new_height/img_height
        left, up, right, bottom = 0, 0, img_width, img_height
        if width_factor <= 1.5:
            crop_width = int((new_width-img_width)/-2)
            left = left + crop_width
            right = right - crop_width
        if height_factor <= 1.5:
            crop_height = int((new_height-img_height)/-2)
            up = up + crop_height
            bottom = bottom - crop_height
        image = image.crop((left, up, right, bottom))
        rescaled_images.append(image)
    return rescaled_images
    
class Diffuser(object):
    @classmethod
    async def create(cls, bot:discord.ext.commands.Bot, config:dict) -> None:
        """
        Creates the worker class, initialising it with the given config.
        """
        self = Diffuser()
        self.bot = bot
        self.config = config
        self.image_queue = asyncio.Queue(maxsize=0)
        self.image_worker = asyncio.create_task(self._image_worker())
        self.model_manager = await Model_Manager.create(
            civitai_token=self.bot.config['civitai_token'], 
            default_model=self.bot.config['default_model'], 
            diffuser_models_path=self.bot.config['diffuser_models_path'], 
            logger=self.bot.logger, 
            owners=self.bot.config['owners']
        )
        self.profiles_path = Path(__file__).parent.parent / 'configs' / 'diffuser_profiles.json'
        try:
            self.user_profiles = await Async_JSON.async_load_json(self.profiles_path)
        except:
            with open(self.profiles_path, 'w') as f:
                json.dump({}, f)
            self.user_profiles = {}
        self.user_queue = []
        return self

    async def _image_worker(self) -> None:
        """
        This asynchronous worker loop is responsible for processing messages one by one from the FIFO image_queue.
        """
        def generate_image(settings_to_pipe:dict) -> None:
            """
            Generate an image using the specified user prompt.
            """
            def get_inputs(pipe_config:dict, compel:Compel):
                """
                Ensures image reproducibility for batching.
                """
                num_images_per_prompt = pipe_config.pop('batch_size')
                pipe_config['num_images_per_prompt'] = num_images_per_prompt
                prompt = pipe_config.pop('prompt')
                pipe_config['prompt_embeds'] = compel.build_conditioning_tensor(prompt)
                negative_prompt = pipe_config.pop('negative_prompt')
                pipe_config['negative_prompt_embeds'] = compel.build_conditioning_tensor(negative_prompt)
                init_image = pipe_config.pop('init_image', None)
                if init_image:
                    pipe_config['image'] = num_images_per_prompt * init_image
                    pipe_config['strength'] = pipe_config.pop('init_strength') 
                seed = pipe_config.pop('seed')
                generator = [torch.Generator("cuda").manual_seed(seed+i) for i in range(num_images_per_prompt)]
                return pipe_config, generator
            
            def nslice(s, n, truncate=False, reverse=False):
                """Splits s into n-sized chunks, optionally reversing the chunks."""
                assert n > 0
                while len(s) >= n:
                    if reverse: yield s[:n][::-1]
                    else: yield s[:n]
                    s = s[n:]
                if len(s) and not truncate:
                    yield s

            ### prepare diffusers pipeline ###
            pipe_config = settings_to_pipe.copy()
            model = pipe_config.pop('model')
            model_info = self.model_manager.get_model_info(model)

            # Use the right pipeline if there is an init
            if settings_to_pipe.get('init_image', None) and 'SD' in model_info['model_pipeline']:
                model_info['model_pipeline'] = model_info['model_pipeline']+'Img2Img'

            pipeline = self.model_manager.get_pipeline(model_info['model_pipeline'])
            pipeline_text2image = pipeline.from_pretrained(
                model_info['path'],
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                cache_dir=self.model_manager.diffuser_models_path,
                safety_checker=None
            )

            # if hires, prepare pipeline by correcting initial dimensions.
            if settings_to_pipe.get('hires_fix', None):
                hires_fix = pipe_config.pop('hires_fix')
                hires_strength = pipe_config.pop('hires_strength')
                model_res = pipeline_text2image.unet.config.sample_size * pipeline_text2image.vae_scale_factor
                while pipe_config['width'] > model_res and pipe_config['height'] > model_res:
                    pipe_config['width'] -= 8
                    pipe_config['height'] -= 8

            # prepare init image
            if settings_to_pipe.get('init_image', None) and 'SD' in model_info['model_pipeline']:
                pipe_config['init_image'] = load_image(pipe_config['init_image'])
                pipe_config['init_image'] = resize_and_crop_centre(pipe_config['init_image'], pipe_config['width'], pipe_config['height'])
                

            pipeline_text2image.to('cuda')
            pipeline_text2image.set_progress_bar_config(disable=True) 

            if model_info['model_pipeline'] in ['SD 1', 'SD 2', 'SDXL']:   
                # Memory and speed optimisation
                pipeline_text2image.enable_attention_slicing()
                pipeline_text2image.enable_vae_slicing()
                pipeline_text2image.enable_model_cpu_offload() 

                # Use the specified scheduler
                scheduler = pipe_config.pop('scheduler')
                pipeline_scheduler = self.model_manager.get_scheduler(scheduler)
                pipeline_text2image.scheduler = pipeline_scheduler.from_config(pipeline_text2image.scheduler.config)
                pipeline_text2image.scheduler.use_karras_sigmas=True

                # Load additional networks
                lora_and_embeds = pipe_config.pop('lora_and_embeds')
                if lora_and_embeds:
                    lora_adapters_and_weights = []
                    for lora_or_embed in lora_and_embeds:
                        index = lora_and_embeds.index(lora_or_embed)
                        lora_or_embed_info = self.model_manager.get_model_info(lora_or_embed.split(':')[0])
                        if lora_or_embed_info['model_type'] == 'LORA':
                            lora_adapters_and_weight = lora_or_embed.split(':')
                            lora = lora_adapters_and_weight[0]
                            if len(lora_adapters_and_weight) == 1:
                                # Add a default weight of 1.0 to the lora and correct it in the response...
                                weight = 1.0
                                settings_to_pipe['lora_and_embeds'].remove(lora_or_embed)
                                settings_to_pipe['lora_and_embeds'].insert(index, lora_or_embed+':1.0')
                            else:
                                # ...Or use the provided weight.
                                try:
                                    weight = float(lora_adapters_and_weight[-1])
                                except:
                                    weight = 1.0
                            pipeline_text2image.load_lora_weights(lora_or_embed_info['path'], adapter_name=lora)
                            lora_adapters_and_weights.append({'lora':lora, 'weight':weight})
                        else:
                            pipeline_text2image.load_textual_inversion(pretrained_model_name_or_path=lora_or_embed_info['path'], token=lora_or_embed_info['embedding_trigger'])
                            pipe_config['prompt'] = lora_or_embed_info['embedding_trigger']+' '+pipe_config['prompt']
                            # Textual Inversion doesn't support weights so correct it for the response
                            if ':' in lora_or_embed:
                                settings_to_pipe['lora_and_embeds'].remove(lora_or_embed)
                                settings_to_pipe['lora_and_embeds'].insert(index, lora_or_embed.split(':')[0])
                    if lora_adapters_and_weights:
                        pipeline_text2image.set_adapters([lora['lora'] for lora in lora_adapters_and_weights], adapter_weights=[lora['weight'] for lora in lora_adapters_and_weights])

            # Diffuse!
            with torch.no_grad():
                textual_inversion_manager = DiffusersTextualInversionManager(pipeline_text2image)
                compel = Compel(tokenizer=pipeline_text2image.tokenizer, text_encoder=pipeline_text2image.text_encoder, textual_inversion_manager=textual_inversion_manager)
                pipe_config, generator = get_inputs(pipe_config, compel)
                if pipe_config.get('image', None):
                    # Split the images into groups of 2 to save vram when doing img2img.
                    sliced_images = nslice(pipe_config['image'], 2)
                    sliced_generators = nslice(generator, 2)
                    for image_block, generator_block in zip(sliced_images, sliced_generators):
                        pipe_config['image'] = image_block
                        pipe_config['num_images_per_prompt'] = len(image_block)
                        images = pipeline_text2image(**pipe_config, generator=generator_block).images
                else:
                    images = pipeline_text2image(**pipe_config, generator=generator).images
                    
                # Optionally upscale completed stable diffusions with another go.
                if settings_to_pipe.get('hires_fix', None):
                    if 'Img2Img' not in model_info['model_pipeline']:
                        pipeline = self.model_manager.get_pipeline(model_info['model_pipeline']+'Img2Img')
                        pipeline_text2image = pipeline(**pipeline_text2image.components)

                    # Split the images into groups of 2 to save vram when upscaling.
                    sliced_images = nslice(images, 2)
                    sliced_generators = nslice(generator, 2)
                    hires_fixed_images = []
                    for image_block, generator_block in zip(sliced_images, sliced_generators):
                        pipe_config['width'] = settings_to_pipe['width']
                        pipe_config['height'] = settings_to_pipe['height']
                        pipe_config['image'] = resize_and_crop_centre(image_block, pipe_config['width'], pipe_config['height'])
                        pipe_config['strength'] = hires_strength
                        pipe_config['num_images_per_prompt'] = len(image_block)
                        hires_image_block = pipeline_text2image(**pipe_config, generator=generator_block).images
                        hires_fixed_images.extend(hires_image_block)
                    images = hires_fixed_images

            pipeline_text2image.to('cpu')

            # convert pil.Image to bytes to send over discord without saving anything to the disk
            image_bytes = []
            for image in images:
                #find a way to put the settings into the image's metadata
                tmp = io.BytesIO()
                image.save(tmp, format='PNG')
                tmp.seek(0)
                image_bytes.append(tmp)

            del pipeline_text2image
            torch.cuda.empty_cache()

            return image_bytes

        async def process_message(context:discord.ext.commands.Context, settings_to_pipe:dict) -> None:
            """
            Prepare and send a response for the user message using their prompt.
            """
            # diffuse inputs
            output = await asyncio.to_thread(generate_image, settings_to_pipe)

            # organise outputs
            files = [
                discord.File(fp=image, filename=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{settings_to_pipe['seed']+i}.png")
                for i, image in enumerate(output)
            ]

            prompt = settings_to_pipe.pop('prompt')
            seed = settings_to_pipe.pop('seed')
            if 'SD' in self.model_manager.get_model_info(settings_to_pipe['model'])['model_pipeline']:
                settings_to_pipe['clip_skip'] = 0 if not settings_to_pipe['clip_skip'] else settings_to_pipe['clip_skip']
                settings_to_pipe['lora_and_embeds'] = None if not settings_to_pipe['lora_and_embeds'] else settings_to_pipe['lora_and_embeds']
            settings_block = ''
            for item in settings_to_pipe:
                if not isinstance(settings_to_pipe[item], list):
                    settings_block += item + ':' + str(settings_to_pipe[item]) + ' '
                else:
                    settings_block += item + ':' + ' '.join(settings_to_pipe[item]) + ' '

            # respond
            await context.reply(
                f"*/bobross prompt:*`{prompt}` *{settings_block.strip()}* with seeds {[seed+i for i in range(settings_to_pipe['batch_size'])]} for {context.author.mention}",
                files=files
            )

        while True:
            ### Worker Loop ###
            try:
                context, settings_to_pipe = await self.image_queue.get()
                await process_message(context, settings_to_pipe)
            except Exception as exception:
                self.bot.logger.error(f"Diffuser encountered an error while generating:\n{exception}")
                embed = Embeds.embed_builder({'title':'', 'description':"~~Something went wrong~~.", 'color':0xE02B2B})
                await context.send(embed=embed, ephemeral=True)
            self.user_queue.remove(context.author.id)
