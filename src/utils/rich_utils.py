from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.table
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def print_metrics_table(report, title: str = "Test metrics") -> None:
    """Print the headline metric table to the console at the end of a test run.

    Lightning's own end-of-test table shows only what went through `self.log`, flat and
    pooled. This shows the same breakdown the PDF's first page does -- overall and per era,
    with the skill scores next to the raw values -- so a multirun over 11 species is
    readable as it scrolls past, without opening 11 PDFs to find the one that went wrong.
    """
    # Imported here rather than at module scope: this module is imported by every entry
    # point, and only the test path needs the metric column definitions.
    from src.plots.panels import format_cell
    from src.plots.report import HEADLINE_COLUMNS

    # Metrics down, scopes across -- the transpose of the PDF's layout. Eleven metrics
    # would not fit across an 80-column terminal, but four scopes always do, so the table
    # stays readable when a multirun scrolls 11 species past.
    scopes = [("overall", report.scalars)] + [
        (str(era["era"]), era.to_dict()) for _, era in report.by_era.iterrows()
    ]

    table = rich.table.Table(title=title, header_style="bold", title_style="bold")
    table.add_column("metric", style="cyan", no_wrap=True)
    for scope, _ in scopes:
        table.add_column(scope, justify="right")

    for key, label in HEADLINE_COLUMNS:
        cells = [format_cell(values.get(key)) or "-" for _, values in scopes]
        table.add_row(label.replace("\n", " "), *cells)

    rich.print(table)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
