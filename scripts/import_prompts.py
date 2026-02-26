#!/usr/bin/env python3
"""
增强版提示词导入脚本

功能设计:
1. Dry-run 模式 - 预览导入内容，不实际执行
2. Force 模式 - 强制重新导入已存在的提示词
3. Selective 模式 - 选择性导入指定提示词
4. Diff 模式 - 显示文件与 MongoDB 版本的差异
5. Export 模式 - 将 MongoDB 提示词导出回文件
6. Backup 模式 - 导入前备份现有提示词
"""
import argparse
import difflib
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# 项目路径处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bz_agent.storage import prompt_store
from utils.logger_config import logger


class PromptImportManager:
    """提示词导入管理器"""

    def __init__(self, prompts_dir: str):
        self.prompts_dir = prompts_dir
        self.prompts_store = prompt_store

    def scan_local_files(self) -> Dict[str, Dict]:
        """扫描本地所有 .md 文件

        Returns:
            {prompt_name: {'path': path, 'content': content, 'size': size, 'modified': timestamp}}
        """
        files = {}
        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.md'):
                name = filename[:-3]  # 去掉 .md
                path = os.path.join(self.prompts_dir, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                files[name] = {
                    'path': path,
                    'content': content,
                    'size': len(content),
                    'modified': os.path.getmtime(path)
                }
        return files

    def get_mongo_prompts(self) -> Dict[str, Dict]:
        """获取 MongoDB 中所有提示词

        Returns:
            {prompt_name: {'version': int, 'template': str, 'active': bool, ...}}
        """
        prompts = {}
        for doc in self.prompts_store.list_prompts(limit=1000):
            name = doc['prompt_name']
            if name not in prompts or doc['version'] > prompts[name]['version']:
                prompts[name] = doc
        return prompts

    def compare_versions(self, name: str, local_content: str, mongo_doc: Dict) -> Dict:
        """比较本地文件与 MongoDB 版本差异

        Returns:
            {
                'exists': bool,
                'identical': bool,
                'diff_lines': List[str],
                'mongo_version': int,
                'mongo_active': bool,
                'local_size': int,
                'mongo_size': int
            }
        """
        mongo_content = mongo_doc.get('template', '')

        result = {
            'exists': True,
            'identical': local_content == mongo_content,
            'mongo_version': mongo_doc.get('version', 0),
            'mongo_active': mongo_doc.get('active', False),
            'local_size': len(local_content),
            'mongo_size': len(mongo_content),
            'local_modified': None
        }

        if not result['identical']:
            # 生成差异对比
            local_lines = local_content.splitlines(keepends=True)
            mongo_lines = mongo_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                mongo_lines, local_lines,
                fromfile=f'MongoDB v{result["mongo_version"]}',
                tofile='Local file',
                lineterm=''
            )
            result['diff_lines'] = list(diff)
        else:
            result['diff_lines'] = []

        return result

    def dry_run(self, specific_names: Optional[List[str]] = None) -> List[Dict]:
        """预览模式 - 不实际导入，只显示将要导入的内容

        Args:
            specific_names: 指定导入的提示词名称列表

        Returns:
            预览结果列表
        """
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - Preview Import")
        logger.info("=" * 60)

        local_files = self.scan_local_files()
        mongo_prompts = self.get_mongo_prompts()

        results = []

        for name in local_files:
            if specific_names and name not in specific_names:
                continue

            file_info = local_files[name]

            if name in mongo_prompts:
                comp = self.compare_versions(name, file_info['content'], mongo_prompts[name])
                status = "UPDATE" if not comp['identical'] else "SKIP"
            else:
                comp = {'exists': False, 'identical': False, 'mongo_version': 0}
                status = "NEW"

            results.append({
                'name': name,
                'status': status,
                'comparison': comp,
                'local_file': file_info
            })

        self._print_preview(results)
        return results

    def import_prompts(
        self,
        force: bool = False,
        specific_names: Optional[List[str]] = None,
        backup_before: bool = False
    ) -> Dict[str, any]:
        """执行导入

        Args:
            force: 强制重新导入已存在的提示词
            specific_names: 指定导入的提示词名称列表
            backup_before: 导入前备份

        Returns:
            导入结果统计
        """
        if backup_before:
            self._backup_existing()

        local_files = self.scan_local_files()
        mongo_prompts = self.get_mongo_prompts()

        stats = {
            'total': 0,
            'new': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'details': []
        }

        for name in local_files:
            if specific_names and name not in specific_names:
                continue

            stats['total'] += 1

            if name in mongo_prompts:
                if force:
                    # 强制更新 - 创建新版本
                    success = self.prompts_store.update_prompt(
                        name, local_files[name]['content']
                    )
                    if success:
                        stats['updated'] += 1
                        stats['details'].append({
                            'name': name,
                            'action': 'UPDATED (force)',
                            'old_version': mongo_prompts[name]['version'],
                            'new_version': mongo_prompts[name]['version'] + 1
                        })
                    else:
                        stats['failed'] += 1
                else:
                    # 跳过已存在
                    stats['skipped'] += 1
                    stats['details'].append({
                        'name': name,
                        'action': 'SKIPPED (already exists)',
                        'version': mongo_prompts[name]['version']
                    })
            else:
                # 新导入
                doc = self.prompts_store.import_from_file(name, self.prompts_dir)
                if doc:
                    stats['new'] += 1
                    stats['details'].append({
                        'name': name,
                        'action': 'IMPORTED',
                        'version': doc['version']
                    })
                else:
                    stats['failed'] += 1

        return stats

    def show_diff(self, name: str) -> Optional[Dict]:
        """显示指定提示词的差异

        Args:
            name: 提示词名称

        Returns:
            差异比较结果
        """
        local_files = self.scan_local_files()
        mongo_prompts = self.get_mongo_prompts()

        if name not in local_files:
            logger.error(f"Local file not found: {name}")
            return None

        if name not in mongo_prompts:
            logger.info(f"Prompt '{name}' does not exist in MongoDB (will be NEW)")
            return {'exists': False}

        comp = self.compare_versions(name, local_files[name]['content'], mongo_prompts[name])
        self._print_diff(name, comp)
        return comp

    def export_to_file(self, name: Optional[str] = None, output_dir: Optional[str] = None) -> int:
        """将 MongoDB 提示词导出到文件

        Args:
            name: 指定导出的提示词名称，None 表示全部导出
            output_dir: 输出目录，None 则覆盖本地文件

        Returns:
            导出的文件数量
        """
        output_dir = output_dir or self.prompts_dir
        os.makedirs(output_dir, exist_ok=True)

        mongo_prompts = self.get_mongo_prompts()
        count = 0

        names = [name] if name else list(mongo_prompts.keys())

        for n in names:
            if n not in mongo_prompts:
                logger.warning(f"Prompt '{n}' not found in MongoDB")
                continue

            doc = mongo_prompts[n]
            output_path = os.path.join(output_dir, f"{n}.md")

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc['template'])

            count += 1
            logger.info(f"Exported: {output_path} (version {doc['version']})")

        return count

    def _backup_existing(self) -> str:
        """备份现有提示词

        Returns:
            备份目录路径
        """
        backup_dir = os.path.join(
            os.path.dirname(self.prompts_dir),
            f"prompts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(backup_dir, exist_ok=True)

        for doc in self.prompts_store.list_prompts(active_only=True):
            name = doc['prompt_name']
            version = doc['version']
            backup_file = os.path.join(backup_dir, f"{name}_v{version}.md")
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(doc['template'])

        logger.info(f"Backup created at: {backup_dir}")
        return backup_dir

    def _print_preview(self, results: List[Dict]) -> None:
        """打印预览结果"""
        logger.info(f"\n{'Name':<20} {'Status':<12} {'Mongo Ver':<10} {'Local Size':<12}")
        logger.info("-" * 60)

        for r in results:
            comp = r['comparison']
            mongo_ver = comp.get('mongo_version', 'N/A')
            local_size = r['local_file']['size']
            logger.info(f"{r['name']:<20} {r['status']:<12} {mongo_ver:<10} {local_size:<12}")

    def _print_diff(self, name: str, comp: Dict) -> None:
        """打印差异"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Diff for: {name}")
        logger.info(f"{'=' * 60}")

        if comp['diff_lines']:
            for line in comp['diff_lines']:
                logger.info(line)
        else:
            logger.info("No differences (identical)")
        logger.info(f"{'=' * 60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='提示词导入/导出管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览导入 (不实际执行)
  python -m scripts.import_prompts --dry-run

  # 强制重新导入所有提示词
  python -m scripts.import_prompts --force

  # 只导入指定提示词
  python -m scripts.import_prompts --name planner supervisor

  # 显示差异
  python -m scripts.import_prompts --diff planner

  # 导出 MongoDB 提示词到文件
  python -m scripts.import_prompts --export

  # 导入前备份
  python -m scripts.import_prompts --backup
        """
    )

    # 模式选择 (互斥)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dry-run', '-d', action='store_true',
                           help='预览模式，不实际导入')
    mode_group.add_argument('--diff', metavar='NAME',
                           help='显示指定提示词与 MongoDB 的差异')
    mode_group.add_argument('--export', '-e', nargs='?', const='all',
                           metavar='NAME',
                           help='导出 MongoDB 提示词到文件 (指定名称或 all)')

    # 导入选项
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制重新导入已存在的提示词')
    parser.add_argument('--name', '-n', nargs='+', metavar='NAME',
                       help='指定导入的提示词名称 (可多个)')
    parser.add_argument('--backup', '-b', action='store_true',
                       help='导入前备份现有提示词')

    # 路径选项
    parser.add_argument('--prompts-dir', '-p',
                       default='bz_agent/prompts',
                       help='提示词文件目录 (默认: bz_agent/prompts)')
    parser.add_argument('--output-dir', '-o',
                       help='导出输出目录')

    args = parser.parse_args()

    manager = PromptImportManager(args.prompts_dir)

    # Diff 模式
    if args.diff:
        manager.show_diff(args.diff)
        return

    # Export 模式
    if args.export is not None:
        name = None if args.export == 'all' else args.export
        count = manager.export_to_file(name, args.output_dir)
        logger.info(f"Exported {count} prompt(s)")
        return

    # Dry-run 模式
    if args.dry_run:
        manager.dry_run(args.name)
        return

    # 正常导入模式 (默认)
    logger.info("=" * 60)
    logger.info("Starting prompt import...")
    logger.info("=" * 60)

    stats = manager.import_prompts(
        force=args.force,
        specific_names=args.name,
        backup_before=args.backup
    )

    logger.info("=" * 60)
    logger.info("Import Summary:")
    logger.info(f"  Total processed: {stats['total']}")
    logger.info(f"  New: {stats['new']}")
    logger.info(f"  Updated: {stats['updated']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
