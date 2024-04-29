import argparse
import asyncio
import json
import pathlib

import aiohttp
import gradio as gr
import requests

conn_url = None
conn_key = None

host_url = "127.0.0.1"

models = []
draft_models = []
loras = []
templates = []

download_task = None

parser = argparse.ArgumentParser(description="TabbyAPI Gradio Loader")
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=7860,
    help="Specify port to host the WebUI on (default 7860)",
)
parser.add_argument(
    "-l", "--listen", action="store_true", help="Share WebUI link via LAN"
)
parser.add_argument(
    "-n",
    "--noauth",
    action="store_true",
    help="Specify TabbyAPI endpoint that has no authorization",
)
parser.add_argument(
    "-s",
    "--share",
    action="store_true",
    help="Share WebUI link remotely via Gradio's built in tunnel",
)
parser.add_argument(
    "-a",
    "--autolaunch",
    action="store_true",
    help="Launch browser after starting WebUI",
)
parser.add_argument(
    "-e",
    "--endpoint_url",
    type=str,
    default="http://localhost:5000",
    help="TabbyAPI endpoint URL (default http://localhost:5000)",
)
parser.add_argument(
    "-k",
    "--admin_key",
    type=str,
    default=None,
    help="TabbyAPI admin key, connect automatically on launch",
)
args = parser.parse_args()
if args.listen:
    host_url = "0.0.0.0"


def read_preset(name):
    if not name:
        raise gr.Error("Please select a preset to load.")
    path = pathlib.Path(f"./presets/{name}.json").resolve()
    with open(path, "r") as openfile:
        data = json.load(openfile)
    gr.Info(f"Preset {name} loaded.")
    return (
        gr.Dropdown(value=data.get("name")),
        gr.Number(value=data.get("max_seq_len")),
        gr.Number(value=data.get("override_base_seq_len")),
        gr.Checkbox(value=data.get("gpu_split_auto")),
        gr.Textbox(value=data.get("gpu_split")),
        gr.Number(value=data.get("rope_scale")),
        gr.Number(value=data.get("rope_alpha")),
        gr.Checkbox(value=data.get("no_flash_attention")),
        gr.Radio(value=data.get("cache_mode")),
        gr.Dropdown(value=data.get("prompt_template")),
        gr.Number(value=data.get("num_experts_per_token")),
        gr.Checkbox(value=data.get("use_cfg")),
        gr.Dropdown(value=data.get("draft_model_name")),
        gr.Number(value=data.get("draft_rope_scale")),
        gr.Number(value=data.get("draft_rope_alpha")),
        gr.Checkbox(value=data.get("fasttensors")),
        gr.Textbox(value=data.get("autosplit_reserve")),
        gr.Number(value=data.get("chunk_size")),
    )


def del_preset(name):
    if not name:
        raise gr.Error("Please select a preset to delete.")
    path = pathlib.Path(f"./presets/{name}.json").resolve()
    path.unlink()
    gr.Info(f"Preset {name} deleted.")
    return get_preset_list()


def write_preset(
    name,
    model_name,
    max_seq_len,
    override_base_seq_len,
    gpu_split_auto,
    gpu_split,
    model_rope_scale,
    model_rope_alpha,
    no_flash_attention,
    cache_mode,
    prompt_template,
    num_experts_per_token,
    use_cfg,
    draft_model_name,
    draft_rope_scale,
    draft_rope_alpha,
    fasttensors,
    autosplit_reserve,
    chunk_size,
):
    if not name:
        raise gr.Error("Please enter a name for your new preset.")
    path = pathlib.Path(f"./presets/{name}.json").resolve()
    data = {
        "name": model_name,
        "max_seq_len": max_seq_len,
        "override_base_seq_len": override_base_seq_len,
        "gpu_split_auto": gpu_split_auto,
        "gpu_split": gpu_split,
        "rope_scale": model_rope_scale,
        "rope_alpha": model_rope_alpha,
        "no_flash_attention": no_flash_attention,
        "cache_mode": cache_mode,
        "prompt_template": prompt_template,
        "num_experts_per_token": num_experts_per_token,
        "use_cfg": use_cfg,
        "draft_model_name": draft_model_name,
        "draft_rope_scale": draft_rope_scale,
        "draft_rope_alpha": draft_rope_alpha,
        "fasttensors": fasttensors,
        "autosplit_reserve": autosplit_reserve,
        "chunk_size": chunk_size,
    }
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=4)
    gr.Info(f"Preset {name} saved.")
    return gr.Textbox(value=None), get_preset_list()


def get_preset_list(raw=False):
    preset_path = pathlib.Path("./presets").resolve()
    preset_list = []
    for path in preset_path.iterdir():
        if path.is_file() and path.name.endswith(".json"):
            preset_list.append(path.stem)
    preset_list.sort(key=str.lower)
    if raw:
        return preset_list
    return gr.Dropdown(choices=[""] + preset_list, value=None)


def connect(api_url, admin_key, silent=False):
    global conn_url
    global conn_key
    global models
    global draft_models
    global loras
    global templates

    if not args.noauth:
        try:
            a = requests.get(
                url=api_url + "/v1/auth/permission", headers={"X-api-key": admin_key}
            )
            a.raise_for_status()
            if a.json().get("permission") != "admin":
                raise ValueError(
                    "The provided authentication key must be an admin key to access the loader's functions."
                )
        except Exception as e:
            raise gr.Error(e)

    try:
        m = requests.get(
            url=api_url + "/v1/model/list", headers={"X-api-key": admin_key}
        )
        m.raise_for_status()
        d = requests.get(
            url=api_url + "/v1/model/draft/list", headers={"X-api-key": admin_key}
        )
        d.raise_for_status()
        lo = requests.get(
            url=api_url + "/v1/lora/list", headers={"X-api-key": admin_key}
        )
        lo.raise_for_status()
        t = requests.get(
            url=api_url + "/v1/template/list", headers={"X-api-key": admin_key}
        )
        t.raise_for_status()
    except Exception as e:
        raise gr.Error(e)

    conn_url = api_url
    conn_key = admin_key

    models = []
    for model in m.json().get("data"):
        models.append(model.get("id"))
    models.sort(key=str.lower)

    draft_models = []
    for draft_model in d.json().get("data"):
        draft_models.append(draft_model.get("id"))
    draft_models.sort(key=str.lower)

    loras = []
    for lora in lo.json().get("data"):
        loras.append(lora.get("id"))
    loras.sort(key=str.lower)

    templates = []
    for template in t.json().get("data"):
        templates.append(template)
    templates.sort(key=str.lower)

    if not silent:
        gr.Info("TabbyAPI connected.")
        return (
            gr.Textbox(value=", ".join(models), visible=True),
            gr.Textbox(value=", ".join(draft_models), visible=True),
            gr.Textbox(value=", ".join(loras), visible=True),
            get_model_list(),
            get_draft_model_list(),
            get_lora_list(),
            get_template_list(),
            get_current_model(),
            get_current_loras(),
        )


def get_model_list():
    return gr.Dropdown(choices=[""] + models, value=None)


def get_draft_model_list():
    return gr.Dropdown(choices=[""] + draft_models, value=None)


def get_lora_list():
    return gr.Dropdown(choices=loras, value=[])


def get_template_list():
    return gr.Dropdown(choices=[""] + templates, value=None)


def get_current_model():
    model_card = requests.get(
        url=conn_url + "/v1/model", headers={"X-api-key": conn_key}
    ).json()
    if not model_card.get("id"):
        return gr.Textbox(value=None)
    params = model_card.get("parameters")
    draft_model_card = params.get("draft")
    model = f'{model_card.get("id")} (context: {params.get("max_seq_len")}, rope scale: {params.get("rope_scale")}, rope alpha: {params.get("rope_alpha")}, cfg: {params.get("use_cfg")})'

    if draft_model_card:
        draft_params = draft_model_card.get("parameters")
        model += f' | {draft_model_card.get("id")} (rope scale: {draft_params.get("rope_scale")}, rope alpha: {draft_params.get("rope_alpha")})'
    return gr.Textbox(value=model)


def get_current_loras():
    lo = requests.get(url=conn_url + "/v1/lora", headers={"X-api-key": conn_key}).json()
    if not lo.get("data"):
        return gr.Textbox(value=None)
    lora_list = lo.get("data")
    loras = []
    for lora in lora_list:
        loras.append(f'{lora.get("id")} (scaling: {lora.get("scaling")})')
    return gr.Textbox(value=", ".join(loras))


def update_loras_table(loras):
    array = []
    for lora in loras:
        array.append(1.0)
    if array:
        return gr.List(
            value=[array],
            col_count=(len(array), "fixed"),
            row_count=(1, "fixed"),
            headers=loras,
            visible=True,
        )
    else:
        return gr.List(value=None, visible=False)


def load_model(
    model_name,
    max_seq_len,
    override_base_seq_len,
    gpu_split_auto,
    gpu_split,
    model_rope_scale,
    model_rope_alpha,
    no_flash_attention,
    cache_mode,
    prompt_template,
    num_experts_per_token,
    use_cfg,
    draft_model_name,
    draft_rope_scale,
    draft_rope_alpha,
    fasttensors,
    autosplit_reserve,
    chunk_size,
):
    if not model_name:
        raise gr.Error("Specify a model to load!")
    gpu_split_parsed = []
    try:
        if gpu_split:
            gpu_split_parsed = [float(i) for i in list(gpu_split.split(","))]
    except ValueError:
        raise gr.Error("Check your GPU split values and ensure they are valid!")
    autosplit_reserve_parsed = []
    try:
        if autosplit_reserve:
            autosplit_reserve_parsed = [
                float(i) for i in list(autosplit_reserve.split(","))
            ]
    except ValueError:
        raise gr.Error("Check your autosplit reserve values and ensure they are valid!")
    if draft_model_name:
        draft_request = {
            "draft_model_name": draft_model_name,
            "draft_rope_scale": draft_rope_scale,
            "draft_rope_alpha": draft_rope_alpha,
        }
    else:
        draft_request = None
    request = {
        "name": model_name,
        "max_seq_len": max_seq_len,
        "override_base_seq_len": override_base_seq_len,
        "gpu_split_auto": gpu_split_auto,
        "gpu_split": gpu_split_parsed,
        "rope_scale": model_rope_scale,
        "rope_alpha": model_rope_alpha,
        "no_flash_attention": no_flash_attention,
        "cache_mode": cache_mode,
        "prompt_template": prompt_template,
        "num_experts_per_token": num_experts_per_token,
        "use_cfg": use_cfg,
        "fasttensors": fasttensors,
        "autosplit_reserve": autosplit_reserve_parsed,
        "chunk_size": chunk_size,
        "draft": draft_request,
    }
    try:
        requests.post(
            url=conn_url + "/v1/model/unload", headers={"X-admin-key": conn_key}
        )
        r = requests.post(
            url=conn_url + "/v1/model/load",
            headers={"X-admin-key": conn_key},
            json=request,
        )
        r.raise_for_status()
        gr.Info("Model successfully loaded.")
        return get_current_model(), get_current_loras()
    except Exception as e:
        raise gr.Error(e)


def load_loras(loras, scalings):
    if not loras:
        raise gr.Error("Specify at least one lora to load!")
    load_list = []
    for index, lora in enumerate(loras):
        try:
            scaling = float(scalings[0][index])
            load_list.append({"name": lora, "scaling": scaling})
        except ValueError:
            raise gr.Error("Check your scaling values and ensure they are valid!")
    request = {"loras": load_list}
    try:
        requests.post(
            url=conn_url + "/v1/lora/unload", headers={"X-admin-key": conn_key}
        )
        r = requests.post(
            url=conn_url + "/v1/lora/load",
            headers={"X-admin-key": conn_key},
            json=request,
        )
        r.raise_for_status()
        gr.Info("Loras successfully loaded.")
        return get_current_model(), get_current_loras()
    except Exception as e:
        raise gr.Error(e)


def unload_model():
    try:
        r = requests.post(
            url=conn_url + "/v1/model/unload", headers={"X-admin-key": conn_key}
        )
        r.raise_for_status()
        gr.Info("Model unloaded.")
        return get_current_model(), get_current_loras()
    except Exception as e:
        raise gr.Error(e)


def unload_loras():
    try:
        r = requests.post(
            url=conn_url + "/v1/lora/unload", headers={"X-admin-key": conn_key}
        )
        r.raise_for_status()
        gr.Info("All loras unloaded.")
        return get_current_model(), get_current_loras()
    except Exception as e:
        raise gr.Error(e)


def toggle_gpu_split(gpu_split_auto):
    if gpu_split_auto:
        return gr.Textbox(value=None, visible=False), gr.Textbox(visible=True)
    else:
        return gr.Textbox(visible=True), gr.Textbox(value=None, visible=False)


def load_template(prompt_template):
    try:
        r = requests.post(
            url=conn_url + "/v1/template/switch",
            headers={"X-admin-key": conn_key},
            json={"name": prompt_template},
        )
        r.raise_for_status()
        gr.Info(f"Prompt template switched to {prompt_template}.")
        return
    except Exception as e:
        raise gr.Error(e)


def unload_template():
    try:
        r = requests.post(
            url=conn_url + "/v1/template/unload", headers={"X-admin-key": conn_key}
        )
        r.raise_for_status()
        gr.Info("Prompt template unloaded.")
        return
    except Exception as e:
        raise gr.Error(e)


async def download(repo_id, revision, repo_type, folder_name, token):
    global download_task
    if download_task:
        return
    request = {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": repo_type.lower(),
        "folder_name": folder_name,
        "token": token,
    }
    try:
        async with aiohttp.ClientSession() as session:
            gr.Info(f"Beginning download of {repo_id}.")
            download_task = asyncio.create_task(
                session.post(
                    url=conn_url + "/v1/download",
                    headers={"X-admin-key": conn_key},
                    json=request,
                )
            )
            r = await download_task
            r.raise_for_status()
            content = await r.json()
            gr.Info(
                f'{repo_type} {repo_id} downloaded to folder: {content.get("download_path")}.'
            )
    except asyncio.CancelledError:
        gr.Info("Download canceled.")
    except Exception as e:
        raise gr.Error(e)
    finally:
        await session.close()
        download_task = None


async def cancel_download():
    global download_task
    if download_task:
        download_task.cancel()


# Auto-attempt connection if admin key is provided
init_model_text = None
init_lora_text = None
if args.admin_key:
    try:
        connect(api_url=args.endpoint_url, admin_key=args.admin_key, silent=True)
        init_model_text = get_current_model().value
        init_lora_text = get_current_loras().value
    except Exception:
        print("Automatic connection failed, continuing to WebUI.")

# Setup UI elements
with gr.Blocks(title="TabbyAPI Gradio Loader") as webui:
    gr.Markdown(
        """
    # TabbyAPI Gradio Loader
    """
    )
    current_model = gr.Textbox(value=init_model_text, label="Current Model:")
    current_loras = gr.Textbox(value=init_lora_text, label="Current Loras:")

    with gr.Tab("Connect to API"):
        connect_btn = gr.Button(value="Connect", variant="primary")
        api_url = gr.Textbox(
            value=args.endpoint_url, label="TabbyAPI Endpoint URL:", interactive=True
        )
        admin_key = gr.Textbox(
            value=args.admin_key, label="Admin Key:", type="password", interactive=True
        )
        model_list = gr.Textbox(
            value=", ".join(models), label="Available Models:", visible=bool(conn_key)
        )
        draft_model_list = gr.Textbox(
            value=", ".join(draft_models),
            label="Available Draft Models:",
            visible=bool(conn_key),
        )
        lora_list = gr.Textbox(
            value=", ".join(loras), label="Available Loras:", visible=bool(conn_key)
        )

    with gr.Tab("Load Model"):
        with gr.Row():
            load_model_btn = gr.Button(value="Load Model", variant="primary")
            unload_model_btn = gr.Button(value="Unload Model", variant="stop")

        with gr.Accordion(open=False, label="Presets"):
            with gr.Row():
                load_preset = gr.Dropdown(
                    choices=[""] + get_preset_list(True),
                    label="Load Preset:",
                    interactive=True,
                )
                save_preset = gr.Textbox(label="Save Preset:", interactive=True)

            with gr.Row():
                load_preset_btn = gr.Button(value="Load Preset", variant="primary")
                del_preset_btn = gr.Button(value="Delete Preset", variant="stop")
                save_preset_btn = gr.Button(value="Save Preset", variant="primary")
                refresh_preset_btn = gr.Button(value="Refresh Presets")

        with gr.Group():
            models_drop = gr.Dropdown(
                choices=[""] + models, label="Select Model:", interactive=True
            )
            with gr.Row():
                max_seq_len = gr.Number(
                    value=lambda: None,
                    label="Max Sequence Length:",
                    precision=0,
                    minimum=1,
                    interactive=True,
                    info="Configured context length to load the model with. If left blank, automatically reads from model config.",
                )
                override_base_seq_len = gr.Number(
                    value=lambda: None,
                    label="Override Base Sequence Length:",
                    precision=0,
                    minimum=1,
                    interactive=True,
                    info="Override the model's 'base' sequence length in config.json. Only relevant when using automatic rope alpha. Leave blank if unsure.",
                )

            with gr.Row():
                model_rope_scale = gr.Number(
                    value=lambda: None,
                    label="Rope Scale:",
                    minimum=1,
                    interactive=True,
                    info="AKA compress_pos_emb or linear rope, used for models trained with modified positional embeddings, such as SuperHoT. If left blank, automatically reads from model config.",
                )
                model_rope_alpha = gr.Number(
                    value=lambda: None,
                    label="Rope Alpha:",
                    minimum=1,
                    interactive=True,
                    info="Factor used for NTK-aware rope scaling. Leave blank for automatic calculation based on your configured max_seq_len and the model's base context length.",
                )

        with gr.Accordion(open=False, label="Speculative Decoding"):
            draft_models_drop = gr.Dropdown(
                choices=[""] + draft_models,
                label="Select Draft Model:",
                interactive=True,
                info="Must share the same tokenizer and vocabulary as the primary model.",
            )
            with gr.Row():
                draft_rope_scale = gr.Number(
                    value=lambda: None,
                    label="Draft Rope Scale:",
                    minimum=1,
                    interactive=True,
                    info="AKA compress_pos_emb or linear rope, used for models trained with modified positional embeddings, such as SuperHoT. If left blank, automatically reads from model config.",
                )
                draft_rope_alpha = gr.Number(
                    value=lambda: None,
                    label="Draft Rope Alpha:",
                    minimum=1,
                    interactive=True,
                    info="Factor used for NTK-aware rope scaling. Leave blank for automatic scaling calculated based on your configured max_seq_len and the model's base context length.",
                )

        with gr.Group():
            with gr.Row():
                cache_mode = gr.Radio(
                    value="FP16",
                    label="Cache Mode:",
                    choices=["Q4", "FP8", "FP16"],
                    interactive=True,
                    info="Q4 and FP8 cache sacrifice some precision to save VRAM compared to full FP16 precision.",
                )
                no_flash_attention = gr.Checkbox(
                    value=False,
                    label="No Flash Attention",
                    interactive=True,
                    info="Disables flash attention, only recommended for old unsupported GPUs.",
                )
                gpu_split_auto = gr.Checkbox(
                    value=True,
                    label="GPU Split Auto",
                    interactive=True,
                    info="Automatically determine how to split model layers between multiple GPUs.",
                )
                use_cfg = gr.Checkbox(
                    value=False,
                    label="Use CFG",
                    interactive=True,
                    info="Enable classifier-free guidance. This requires additional VRAM for the negative prompt cache.",
                )
                fasttensors = gr.Checkbox(
                    value=False,
                    label="Use Fasttensors",
                    interactive=True,
                    info="Enable to possibly increase model loading speeds on some systems.",
                )

            gpu_split = gr.Textbox(
                label="GPU Split:",
                placeholder="20.6,24",
                visible=False,
                interactive=True,
                info="Amount of VRAM TabbyAPI will be allowed to use on each GPU. List of numbers separated by commas, in gigabytes.",
            )
            autosplit_reserve = gr.Textbox(
                label="Auto-split Reserve:",
                placeholder="96",
                interactive=True,
                info="Amount of VRAM to keep reserved on each GPU when using auto split. List of numbers separated by commas, in megabytes.",
            )
            with gr.Row():
                num_experts_per_token = gr.Number(
                    value=lambda: None,
                    label="Number of experts per token (MoE only):",
                    precision=0,
                    minimum=1,
                    interactive=True,
                    info="Number of experts to use for simultaneous inference in mixture of experts. If left blank, automatically reads from model config.",
                )
                chunk_size = gr.Number(
                    value=lambda: None,
                    label="Chunk Size:",
                    precision=0,
                    minimum=1,
                    interactive=True,
                    info="The number of prompt tokens to ingest at a time. A lower value reduces VRAM usage at the cost of ingestion speed.",
                )

        with gr.Accordion(open=True, label="Prompt Templates"):
            prompt_template = gr.Dropdown(
                choices=[""] + templates,
                value="",
                label="Prompt Template:",
                allow_custom_value=True,
                interactive=True,
                info="Jinja2 prompt template to be used for the chat completions endpoint.",
            )
            with gr.Row():
                load_template_btn = gr.Button(value="Load Template", variant="primary")
                unload_template_btn = gr.Button(value="Unload Template", variant="stop")

    with gr.Tab("Load Loras"):
        with gr.Row():
            load_loras_btn = gr.Button(value="Load Loras", variant="primary")
            unload_loras_btn = gr.Button(value="Unload All Loras", variant="stop")

        loras_drop = gr.Dropdown(
            label="Select Loras:",
            choices=loras,
            multiselect=True,
            interactive=True,
            info="Select one or more loras to load, specify individual lora weights in the box that appears below (default 1.0).",
        )
        loras_table = gr.List(
            label="Lora Scaling:",
            visible=False,
            datatype="number",
            type="array",
            interactive=True,
        )

    with gr.Tab("HF Downloader"):
        with gr.Row():
            download_btn = gr.Button(value="Download", variant="primary")
            cancel_download_btn = gr.Button(value="Cancel", variant="stop")

        with gr.Row():
            repo_id = gr.Textbox(
                label="Repo ID:",
                interactive=True,
                info="Provided in the format <user/organization name>/<repo name>.",
            )
            revision = gr.Textbox(
                label="Revision/Branch:",
                interactive=True,
                info="Name of the revision/branch of the repository to download.",
            )

        with gr.Row():
            repo_type = gr.Dropdown(
                choices=["Model", "Lora"],
                value="Model",
                label="Repo Type:",
                interactive=True,
                info="Specify whether the repository contains a model or lora.",
            )
            folder_name = gr.Textbox(
                label="Folder Name:",
                interactive=True,
                info="Name to use for the local downloaded copy of the repository.",
            )

        with gr.Row():
            token = gr.Textbox(
                label="HF Access Token:",
                type="password",
                info="Provide HF access token to download from private/gated repositories.",
            )

    # Define event listeners
    # Connection tab
    connect_btn.click(
        fn=connect,
        inputs=[api_url, admin_key],
        outputs=[
            model_list,
            draft_model_list,
            lora_list,
            models_drop,
            draft_models_drop,
            loras_drop,
            prompt_template,
            current_model,
            current_loras,
        ],
    )

    # Model tab
    load_preset_btn.click(
        fn=read_preset,
        inputs=load_preset,
        outputs=[
            models_drop,
            max_seq_len,
            override_base_seq_len,
            gpu_split_auto,
            gpu_split,
            model_rope_scale,
            model_rope_alpha,
            no_flash_attention,
            cache_mode,
            prompt_template,
            num_experts_per_token,
            use_cfg,
            draft_models_drop,
            draft_rope_scale,
            draft_rope_alpha,
            fasttensors,
            autosplit_reserve,
            chunk_size,
        ],
    )
    del_preset_btn.click(fn=del_preset, inputs=load_preset, outputs=load_preset)
    save_preset_btn.click(
        fn=write_preset,
        inputs=[
            save_preset,
            models_drop,
            max_seq_len,
            override_base_seq_len,
            gpu_split_auto,
            gpu_split,
            model_rope_scale,
            model_rope_alpha,
            no_flash_attention,
            cache_mode,
            prompt_template,
            num_experts_per_token,
            use_cfg,
            draft_models_drop,
            draft_rope_scale,
            draft_rope_alpha,
            fasttensors,
            autosplit_reserve,
            chunk_size,
        ],
        outputs=[save_preset, load_preset],
    )
    refresh_preset_btn.click(fn=get_preset_list, outputs=load_preset)

    gpu_split_auto.change(
        fn=toggle_gpu_split,
        inputs=gpu_split_auto,
        outputs=[gpu_split, autosplit_reserve],
    )
    unload_model_btn.click(fn=unload_model, outputs=[current_model, current_loras])
    load_model_btn.click(
        fn=load_model,
        inputs=[
            models_drop,
            max_seq_len,
            override_base_seq_len,
            gpu_split_auto,
            gpu_split,
            model_rope_scale,
            model_rope_alpha,
            no_flash_attention,
            cache_mode,
            prompt_template,
            num_experts_per_token,
            use_cfg,
            draft_models_drop,
            draft_rope_scale,
            draft_rope_alpha,
            fasttensors,
            autosplit_reserve,
            chunk_size,
        ],
        outputs=[current_model, current_loras],
    )
    load_template_btn.click(fn=load_template, inputs=prompt_template)
    unload_template_btn.click(fn=unload_template)

    # Loras tab
    loras_drop.change(update_loras_table, inputs=loras_drop, outputs=loras_table)
    unload_loras_btn.click(fn=unload_loras, outputs=[current_model, current_loras])
    load_loras_btn.click(
        fn=load_loras,
        inputs=[loras_drop, loras_table],
        outputs=[current_model, current_loras],
    )

    # HF Downloader tab
    download_btn.click(
        fn=download, inputs=[repo_id, revision, repo_type, folder_name, token]
    )
    cancel_download_btn.click(fn=cancel_download)

webui.launch(
    inbrowser=args.autolaunch,
    show_api=False,
    server_name=host_url,
    server_port=args.port,
    share=args.share,
)
