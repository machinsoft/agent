from __future__ import annotations

from .ops import (
    GitToolingError,
    ensure_git_repository,
    resolve_head,
    resolve_repository_root,
    normalize_relative_path,
    apply_repo_prefix_to_force_include,
    repo_subdir,
    run_git_for_status,
    run_git_for_stdout,
    run_git_for_stdout_all,
)
from .branch import merge_base_with_head
from .info import git_diff_to_remote
from .apply import (
    ApplyGitRequest,
    ApplyGitResult,
    apply_git_patch,
    extract_paths_from_patch,
    stage_paths,
    parse_git_apply_output,
)
from .platform import create_symlink
from .ghost import (
    GhostCommit,
    GhostSnapshotConfig,
    GhostSnapshotReport,
    LargeUntrackedDir,
    IgnoredUntrackedFile,
    CreateGhostCommitOptions,
    RestoreGhostCommitOptions,
    create_ghost_commit,
    capture_ghost_snapshot_report,
    restore_ghost_commit,
    restore_ghost_commit_with_options,
    restore_to_commit,
)
