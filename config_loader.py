import json
import os


# 项目根目录 = config.json 所在目录（与脚本运行时的 cwd 无关）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """加载 config.json，返回完整配置字典。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"未找到配置文件 {config_path}，请确保 config.json 存在于项目根目录。"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_project_root(config_path=DEFAULT_CONFIG_PATH):
    """返回 config.json 所在目录，作为所有相对路径的基准。"""
    return os.path.dirname(os.path.abspath(config_path))


def resolve_path(relative_path, config_path=DEFAULT_CONFIG_PATH, mkdir=False):
    """
    将 config 中的相对路径解析为绝对路径（基准 = 项目根目录）。
    已是绝对路径则原样返回；mkdir=True 时自动创建父目录。
    """
    if not relative_path:
        return relative_path

    if os.path.isabs(relative_path):
        resolved = relative_path
    else:
        resolved = os.path.join(get_project_root(config_path), relative_path)

    if mkdir:
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)

    return resolved


def get_section(config, section, defaults=None):
    """读取某个配置分区，忽略 _comments / _file_description 等元数据键。"""
    defaults = defaults or {}
    raw = config.get(section, {})
    return {k: v for k, v in raw.items() if not k.startswith("_")} or defaults


def get_paths(config, config_path=DEFAULT_CONFIG_PATH, mkdir=False):
    """读取 paths 分区：目录/文件路径解析为绝对路径，模板字符串保持原样。"""
    defaults = {
        "simulated_csv": "shopee_cat_food_simulated.csv",
        "demand_curve_plot": "demand_elasticity_curve.png",
        "tensorboard_dir": "runs",
        "tensorboard_run_prefix": "SAC_Shopee",
        "models_dir": "models",
        "best_model_template": "sac_actor_best_{difficulty}.pth",
        "final_model_name": "sac_actor_final.pth",
        "train_log": "train.out",
        "eval_decision_curves": "shopee_sac_decision_curves.png",
        "baseline_comparison_csv": "baseline_comparison.csv",
    }
    raw = {**defaults, **get_section(config, "paths")}

    # 仅以下键需要相对项目根目录解析
    path_keys = {
        "simulated_csv",
        "demand_curve_plot",
        "tensorboard_dir",
        "models_dir",
        "train_log",
        "eval_decision_curves",
        "baseline_comparison_csv",
    }
    resolved = dict(raw)
    for key in path_keys:
        resolved[key] = resolve_path(raw[key], config_path, mkdir=mkdir)
    return resolved


def model_path(paths, difficulty=None, final=False):
    """根据 paths 配置生成模型文件绝对路径。"""
    models_dir = paths["models_dir"]
    if final:
        return os.path.join(models_dir, paths["final_model_name"])
    template = paths["best_model_template"]
    return os.path.join(models_dir, template.format(difficulty=difficulty))


def update_section(config, section, updates):
    """合并更新到指定分区，保留 _comments 等元数据键。"""
    existing = config.setdefault(section, {})
    for key, value in updates.items():
        if not key.startswith("_"):
            existing[key] = value
    return config


def save_config(config, config_path=DEFAULT_CONFIG_PATH):
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
