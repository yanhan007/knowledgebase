"""端到端测试脚本

模拟完整问答流程：
1. 上传文档到知识库
2. 进行多轮问答
3. 验证答案准确性
4. 性能基准测试
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.qa_engine import QAEngine, QAResult, setup_logging
from config import settings


# ==================== 测试数据 ====================


@dataclass
class TestCase:
    """测试用例"""
    question: str
    expected_keywords: List[str]  # 答案应包含的关键词
    category: str = "general"  # 问题类别


@dataclass
class BenchmarkResult:
    """性能基准结果"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time_ms: float = 0
    query_times_ms: List[float] = field(default_factory=list)
    retrieval_times_ms: List[float] = field(default_factory=list)
    llm_times_ms: List[float] = field(default_factory=list)
    first_token_times_ms: List[float] = field(default_factory=list)

    @property
    def avg_query_time_ms(self) -> float:
        return self.total_time_ms / self.total_queries if self.total_queries else 0

    @property
    def p50_query_time_ms(self) -> float:
        if not self.query_times_ms:
            return 0
        sorted_times = sorted(self.query_times_ms)
        idx = len(sorted_times) // 2
        return sorted_times[idx]

    @property
    def p95_query_time_ms(self) -> float:
        if not self.query_times_ms:
            return 0
        sorted_times = sorted(self.query_times_ms)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_query_time_ms(self) -> float:
        if not self.query_times_ms:
            return 0
        sorted_times = sorted(self.query_times_ms)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def summary(self) -> str:
        return (
            f"查询总数: {self.total_queries}\n"
            f"成功: {self.successful_queries}, 失败: {self.failed_queries}\n"
            f"成功率: {self.successful_queries / self.total_queries * 100:.1f}%\n"
            f"总耗时: {self.total_time_ms:.0f}ms\n"
            f"平均耗时: {self.avg_query_time_ms:.0f}ms\n"
            f"P50: {self.p50_query_time_ms:.0f}ms\n"
            f"P95: {self.p95_query_time_ms:.0f}ms\n"
            f"P99: {self.p99_query_time_ms:.0f}ms"
        )


# ==================== 测试文档生成 ====================


def create_test_documents() -> List[Tuple[str, str]]:
    """创建测试文档（文件路径，内容）"""
    docs = []

    # GIS 基础知识文档
    docs.append((
        "gis_basics.txt",
        """地理信息系统（GIS）基础

地理信息系统（Geographic Information System，GIS）是一种用于采集、存储、管理、分析和显示地理空间数据的计算机系统。
GIS 结合了地理学、地图学、遥感技术和计算机科学，广泛应用于城市规划、环境监测、资源管理、交通规划等领域。

GIS 的核心功能包括：
1. 数据采集：通过 GPS、遥感、测量等方式获取地理数据
2. 数据存储：使用空间数据库管理矢量和栅格数据
3. 数据分析：空间查询、缓冲区分析、叠加分析、网络分析等
4. 数据可视化：地图制作、三维可视化、专题图等

常见的 GIS 软件包括 ArcGIS、QGIS、MapGIS 等。
开源 GIS 工具 QGIS 是目前最流行的开源桌面 GIS 软件，支持多种数据格式和插件扩展。
    """,
    ))

    # 测绘学文档
    docs.append((
        "surveying.txt",
        """测绘学概论

测绘学是研究地球形状、大小和重力场，测定地面点的平面位置和高程的科学。
测绘学的主要分支包括：

1. 大地测量学：研究地球形状和大小，建立大地控制网
2. 摄影测量学：利用摄影影像测定物体的形状和位置
3. 工程测量学：为工程建设提供测量服务
4. 地图制图学：研究地图编制的理论和方法
5. 海洋测绘：测量海洋地形和海底地貌

现代测绘技术：
- GNSS（全球导航卫星系统）：包括 GPS、北斗、GLONASS、Galileo
- LiDAR（激光雷达）：通过激光扫描获取高精度三维点云
- InSAR（合成孔径雷达干涉）：监测地表形变
- 无人机航测：快速获取高分辨率影像

我国自主建设的北斗卫星导航系统（BDS）已全球组网，提供高精度定位服务。
    """,
    ))

    # 遥感技术文档
    docs.append((
        "remote_sensing.txt",
        """遥感技术与应用

遥感（Remote Sensing）是指不直接接触物体，通过传感器获取其信息的技术。
遥感技术主要利用电磁波（可见光、红外、微波等）探测地物特性。

遥感平台分类：
1. 地面遥感：车载、船载传感器
2. 航空遥感：飞机、无人机搭载传感器
3. 航天遥感：卫星搭载传感器

常用遥感卫星：
- Landsat 系列（美国）：30m 分辨率，免费开放
- Sentinel 系列（欧洲）：10m 分辨率，免费开放
- 高分系列（中国）：亚米级分辨率
- WorldView（美国）：0.3m 分辨率

遥感应用领域：
1. 土地利用/覆盖变化监测
2. 植被指数（NDVI）计算
3. 城市热岛效应分析
4. 洪水灾害监测
5. 农作物长势监测

遥感影像处理包括：辐射校正、几何校正、大气校正、图像增强、分类等步骤。
    """,
    ))

    # 空间分析文档
    docs.append((
        "spatial_analysis.txt",
        """空间分析方法

空间分析是 GIS 的核心功能之一，用于分析地理空间数据的模式、关系和趋势。

基本空间分析方法：

1. 缓冲区分析（Buffer Analysis）
   - 围绕点、线、面要素创建指定距离的区域
   - 应用：确定保护区范围、影响区域分析

2. 叠加分析（Overlay Analysis）
   - 将多个图层进行空间叠加，生成新的要素
   - 类型：相交（Intersect）、并集（Union）、裁剪（Clip）

3. 网络分析（Network Analysis）
   - 基于网络数据集进行路径分析、服务区分析
   - 应用：物流配送、交通规划

4. 核密度分析（Kernel Density）
   - 计算点要素在周围区域的密度
   - 应用：犯罪热点分析、设施分布分析

5. 空间插值（Spatial Interpolation）
   - 由离散点数据推测连续表面
   - 方法：反距离权重（IDW）、克里金（Kriging）、样条函数

6. 地形分析
   - 数字高程模型（DEM）生成
   - 坡度、坡向、山体阴影计算
   - 流域分析、可视域分析

空间分析在城市规划、环境评估、应急管理等领域有广泛应用。
    """,
    ))

    return docs


def create_test_files(temp_dir: str) -> List[str]:
    """创建测试文件并返回路径列表"""
    docs = create_test_documents()
    file_paths = []

    for filename, content in docs:
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        file_paths.append(filepath)

    return file_paths


# ==================== 测试用例 ====================


def get_test_cases() -> List[TestCase]:
    """获取测试用例"""
    return [
        # 基础概念测试
        TestCase(
            question="什么是 GIS？",
            expected_keywords=["地理信息系统", "采集", "存储", "管理", "分析"],
            category="基础概念",
        ),
        TestCase(
            question="测绘学是什么？",
            expected_keywords=["地球", "形状", "测量", "平面位置", "高程"],
            category="基础概念",
        ),
        TestCase(
            question="遥感技术的原理是什么？",
            expected_keywords=["电磁波", "传感器", "不直接接触"],
            category="基础概念",
        ),
        # 细节查询测试
        TestCase(
            question="GIS 有哪些核心功能？",
            expected_keywords=["数据采集", "数据存储", "数据分析", "数据可视化"],
            category="细节查询",
        ),
        TestCase(
            question="常用的遥感卫星有哪些？",
            expected_keywords=["Landsat", "Sentinel", "高分", "WorldView"],
            category="细节查询",
        ),
        TestCase(
            question="空间分析有哪些方法？",
            expected_keywords=["缓冲区", "叠加", "网络", "插值"],
            category="细节查询",
        ),
        # 应用场景测试
        TestCase(
            question="遥感技术有哪些应用？",
            expected_keywords=["土地利用", "植被", "监测", "灾害"],
            category="应用场景",
        ),
        TestCase(
            question="现代测绘技术包括哪些？",
            expected_keywords=["GNSS", "LiDAR", "北斗", "无人机"],
            category="应用场景",
        ),
        # 综合分析测试
        TestCase(
            question="GIS 和遥感有什么关系？",
            expected_keywords=["GIS", "遥感", "数据"],
            category="综合分析",
        ),
        TestCase(
            question="如何进行空间插值分析？",
            expected_keywords=["插值", "离散", "连续", "IDW", "克里金"],
            category="综合分析",
        ),
    ]


# ==================== 测试执行 ====================


class E2ETestRunner:
    """端到端测试运行器"""

    def __init__(
        self,
        llm_provider: str = "deepseek",
        llm_model: str = "deepseek-chat",
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.engine = QAEngine(
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        self.benchmark = BenchmarkResult()
        self.test_results: List[Dict] = []

    def log(self, message: str, level: str = "info"):
        """日志输出"""
        if not self.verbose:
            return

        prefix = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
            "header": "📋",
        }.get(level, "  ")

        print(f"{prefix} {message}")

    def run_document_ingestion(self, file_paths: List[str]) -> bool:
        """步骤 1：文档导入测试"""
        self.log("=" * 60, "header")
        self.log("步骤 1：文档导入测试", "header")
        self.log("=" * 60, "header")

        total_chunks = 0
        start_time = time.time()

        for filepath in file_paths:
            filename = os.path.basename(filepath)
            try:
                result = self.engine.ingest_document(filepath)
                # ingest_document 返回 dict: {"success": True, "chunks": N, ...}
                if result.get("success"):
                    chunks = result.get("chunks", 0)
                    total_chunks += chunks
                    self.log(f"导入 {filename}: {chunks} 个文本块 ({result.get('elapsed_ms', 0):.0f}ms)", "success")
                else:
                    self.log(f"导入 {filename} 失败: {result.get('error', '未知错误')}", "error")
                    return False
            except Exception as e:
                self.log(f"导入 {filename} 失败: {e}", "error")
                return False

        elapsed_ms = (time.time() - start_time) * 1000
        self.log(f"\n导入完成: {len(file_paths)} 个文件, {total_chunks} 个文本块", "success")
        self.log(f"导入耗时: {elapsed_ms:.0f}ms", "info")

        # 健康检查
        health = self.engine.health_check()
        all_ok = all(health.values())
        status = "正常" if all_ok else "异常"
        self.log(f"系统健康状态: {status}", "success" if all_ok else "warning")
        for component, ok in health.items():
            self.log(f"  - {component}: {'✓' if ok else '✗'}", "info")

        return True

    def run_qa_tests(self, test_cases: List[TestCase]) -> Tuple[int, int]:
        """步骤 2：多轮问答测试"""
        self.log("\n" + "=" * 60, "header")
        self.log("步骤 2：多轮问答测试", "header")
        self.log("=" * 60, "header")

        passed = 0
        failed = 0

        for i, test_case in enumerate(test_cases, 1):
            self.log(f"\n测试 {i}/{len(test_cases)} [{test_case.category}]", "info")
            self.log(f"问题: {test_case.question}", "info")

            start_time = time.time()
            result = self.engine.ask(test_case.question)
            elapsed_ms = (time.time() - start_time) * 1000

            # 验证答案
            is_passed, match_details = self._verify_answer(
                result.answer, test_case.expected_keywords
            )

            if is_passed:
                passed += 1
                self.log(f"通过 ({elapsed_ms:.0f}ms)", "success")
            else:
                failed += 1
                self.log(f"失败 ({elapsed_ms:.0f}ms)", "error")
                self.log(f"匹配详情: {match_details}", "warning")

            # 记录结果
            self.test_results.append({
                "question": test_case.question,
                "category": test_case.category,
                "passed": is_passed,
                "elapsed_ms": elapsed_ms,
                "answer_length": len(result.answer),
                "citations_count": len(result.rag_result.citations) if result.rag_result else 0,
                "match_details": match_details,
            })

            # 显示答案摘要
            if self.verbose:
                answer_preview = result.answer[:150].replace("\n", " ")
                if len(result.answer) > 150:
                    answer_preview += "..."
                self.log(f"答案: {answer_preview}", "info")

        return passed, failed

    def _verify_answer(
        self, answer: str, expected_keywords: List[str]
    ) -> Tuple[bool, str]:
        """验证答案是否包含预期关键词"""
        answer_lower = answer.lower()
        matched = []
        missing = []

        for keyword in expected_keywords:
            if keyword.lower() in answer_lower:
                matched.append(keyword)
            else:
                missing.append(keyword)

        # 至少匹配 60% 的关键词
        match_ratio = len(matched) / len(expected_keywords) if expected_keywords else 0
        is_passed = match_ratio >= 0.6

        details = f"匹配: {len(matched)}/{len(expected_keywords)} ({match_ratio:.0%})"
        if missing:
            details += f" | 缺失: {', '.join(missing)}"

        return is_passed, details

    def run_benchmark(self, num_queries: int = 10) -> BenchmarkResult:
        """步骤 3：性能基准测试"""
        self.log("\n" + "=" * 60, "header")
        self.log("步骤 3：性能基准测试", "header")
        self.log("=" * 60, "header")

        benchmark_questions = [
            "什么是 GIS？",
            "测绘学的主要分支有哪些？",
            "遥感技术的应用领域？",
            "空间分析有哪些方法？",
            "北斗系统的特点是什么？",
            "GIS 的核心功能是什么？",
            "常用的遥感卫星有哪些？",
            "缓冲区分析的应用场景？",
            "数字高程模型的作用？",
            "遥感影像处理的步骤？",
        ]

        # 确保足够的问题
        questions = []
        while len(questions) < num_queries:
            questions.extend(benchmark_questions)
        questions = questions[:num_queries]

        self.log(f"执行 {num_queries} 次查询...\n", "info")

        benchmark = BenchmarkResult()
        benchmark.total_queries = num_queries

        total_start = time.time()

        for i, question in enumerate(questions, 1):
            try:
                start_time = time.time()
                result = self.engine.ask(question)
                elapsed_ms = (time.time() - start_time) * 1000

                benchmark.successful_queries += 1
                benchmark.query_times_ms.append(elapsed_ms)

                # 记录各阶段耗时
                if result.metrics:
                    if result.metrics.retrieval_ms:
                        benchmark.retrieval_times_ms.append(result.metrics.retrieval_ms)
                    if result.metrics.llm_total_ms:
                        benchmark.llm_times_ms.append(result.metrics.llm_total_ms)
                    if result.metrics.llm_first_token_ms:
                        benchmark.first_token_times_ms.append(
                            result.metrics.llm_first_token_ms
                        )

                # 进度显示
                status = "✓"
                self.log(f"[{i}/{num_queries}] {status} {elapsed_ms:.0f}ms - {question[:30]}...", "info")

            except Exception as e:
                benchmark.failed_queries += 1
                self.log(f"[{i}/{num_queries}] ✗ 失败 - {question[:30]}...: {e}", "error")

        benchmark.total_time_ms = (time.time() - total_start) * 1000

        self.log("\n" + "-" * 40, "info")
        self.log("基准测试结果:", "header")
        self.log(benchmark.summary(), "info")

        # 分阶段统计
        if benchmark.retrieval_times_ms:
            avg_retrieval = sum(benchmark.retrieval_times_ms) / len(benchmark.retrieval_times_ms)
            self.log(f"\n检索平均耗时: {avg_retrieval:.0f}ms", "info")

        if benchmark.llm_times_ms:
            avg_llm = sum(benchmark.llm_times_ms) / len(benchmark.llm_times_ms)
            self.log(f"LLM 平均耗时: {avg_llm:.0f}ms", "info")

        if benchmark.first_token_times_ms:
            avg_first_token = sum(benchmark.first_token_times_ms) / len(benchmark.first_token_times_ms)
            self.log(f"首 Token 平均耗时: {avg_first_token:.0f}ms", "info")

        return benchmark

    def run_stream_test(self) -> bool:
        """步骤 4：流式输出测试"""
        self.log("\n" + "=" * 60, "header")
        self.log("步骤 4：流式输出测试", "header")
        self.log("=" * 60, "header")

        question = "请详细介绍一下 GIS 的核心功能和应用"
        self.log(f"问题: {question}\n", "info")

        try:
            start_time = time.time()
            first_token_time = None
            full_answer = ""
            chunk_count = 0

            for chunk, result in self.engine.ask_stream(question):
                if chunk and not first_token_time:
                    first_token_time = (time.time() - start_time) * 1000
                    self.log(f"首 Token 延迟: {first_token_time:.0f}ms", "info")

                if chunk:
                    chunk_count += 1
                    full_answer += chunk
                    # 简单进度显示
                    if chunk_count % 10 == 0:
                        print(".", end="", flush=True)

            total_time = (time.time() - start_time) * 1000
            print()  # 换行

            self.log(f"\n流式输出完成:", "success")
            self.log(f"  - 总耗时: {total_time:.0f}ms", "info")
            self.log(f"  - Token 数: {chunk_count}", "info")
            self.log(f"  - 答案长度: {len(full_answer)} 字符", "info")

            return True

        except Exception as e:
            self.log(f"流式输出失败: {e}", "error")
            return False

    def generate_report(self, output_path: str = None):
        """生成测试报告"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engine_config": {
                "llm_provider": self.engine.llm.config.provider.value,
                "llm_model": self.engine.llm.config.model,
            },
            "test_results": self.test_results,
            "benchmark": {
                "total_queries": self.benchmark.total_queries,
                "successful_queries": self.benchmark.successful_queries,
                "failed_queries": self.benchmark.failed_queries,
                "total_time_ms": self.benchmark.total_time_ms,
                "avg_query_time_ms": self.benchmark.avg_query_time_ms,
                "p50_query_time_ms": self.benchmark.p50_query_time_ms,
                "p95_query_time_ms": self.benchmark.p95_query_time_ms,
                "p99_query_time_ms": self.benchmark.p99_query_time_ms,
            },
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.log(f"\n报告已保存: {output_path}", "success")

        return report

    def run_all(self, benchmark_queries: int = 10) -> bool:
        """运行全部测试"""
        self.log("\n" + "=" * 60, "header")
        self.log("知识库问答系统 - 端到端测试", "header")
        self.log("=" * 60, "header")

        # 创建临时目录和测试文档
        with tempfile.TemporaryDirectory() as temp_dir:
            self.log(f"临时目录: {temp_dir}", "info")

            # 1. 创建测试文档
            file_paths = create_test_files(temp_dir)
            self.log(f"创建 {len(file_paths)} 个测试文档", "info")

            # 2. 文档导入
            if not self.run_document_ingestion(file_paths):
                self.log("文档导入失败，测试终止", "error")
                return False

            # 3. 多轮问答测试
            test_cases = get_test_cases()
            passed, failed = self.run_qa_tests(test_cases)

            self.log(f"\n问答测试结果: {passed} 通过, {failed} 失败",
                    "success" if failed == 0 else "warning")

            # 4. 性能基准测试
            self.benchmark = self.run_benchmark(benchmark_queries)

            # 5. 流式输出测试
            stream_ok = self.run_stream_test()

            # 6. 生成报告
            report_path = os.path.join(project_root, "tests", "e2e_report.json")
            self.generate_report(report_path)

            # 最终总结
            self.log("\n" + "=" * 60, "header")
            self.log("测试总结", "header")
            self.log("=" * 60, "header")
            self.log(f"问答测试: {passed}/{passed + failed} 通过",
                    "success" if failed == 0 else "warning")
            self.log(f"性能测试: {self.benchmark.successful_queries}/{self.benchmark.total_queries} 成功",
                    "success" if self.benchmark.failed_queries == 0 else "warning")
            self.log(f"流式测试: {'通过' if stream_ok else '失败'}",
                    "success" if stream_ok else "error")

            overall_ok = (failed == 0 and
                         self.benchmark.failed_queries == 0 and
                         stream_ok)
            self.log(f"\n总体结果: {'✅ 全部通过' if overall_ok else '❌ 存在失败'}",
                    "success" if overall_ok else "error")

            return overall_ok


# ==================== 命令行入口 ====================


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="知识库问答系统端到端测试")
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["ollama", "openai", "deepseek"],
        help="LLM 提供商 (默认: deepseek)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="LLM 模型名称 (默认: deepseek-chat)",
    )
    parser.add_argument(
        "--benchmark-queries",
        type=int,
        default=10,
        help="基准测试查询次数 (默认: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="详细输出",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: WARNING)",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(level=args.log_level)

    # 运行测试
    runner = E2ETestRunner(
        llm_provider=args.provider,
        llm_model=args.model,
        verbose=args.verbose,
    )

    success = runner.run_all(benchmark_queries=args.benchmark_queries)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
