import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from typing import List, Dict, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

from data_structures import (
    Course, Classroom, TimeSlot, Schedule, ScheduleAssignment
)
from conflict_detector import ConflictReport, Conflict
from genetic_scheduler import GAHistory


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _get_distinct_colors(n: int) -> np.ndarray:
    cmap = plt.cm.get_cmap('tab20', n)
    colors = cmap(np.linspace(0, 1, n))
    return colors


def _get_timeslot_coords(timeslot: TimeSlot) -> Tuple[int, int]:
    return timeslot.day, timeslot.period


def visualize_timetable(
    schedule: Schedule,
    expanded_courses: List[Course],
    assignments: List[ScheduleAssignment],
    conflicting_courses: Optional[Set[int]] = None,
    adjusted_courses: Optional[Set[int]] = None,
    title: str = "课程安排表",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
):
    day_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    period_names = ["第1-2节", "第3-4节", "第5-6节", "第7-8节", "第9-10节"]

    num_days = 5
    num_periods = 5

    course_map = {c.course_id: c for c in expanded_courses}
    classroom_map = schedule.get_classroom_map()
    timeslot_map = schedule.get_timeslot_map()

    colors = _get_distinct_colors(len(expanded_courses))
    color_map = {c.course_id: colors[i] for i, c in enumerate(expanded_courses)}

    fig, ax = plt.subplots(figsize=figsize)

    cell_width = 1.0
    cell_height = 1.0

    for d in range(num_days + 1):
        ax.axvline(x=d * cell_width, color='black', linewidth=1)
    for p in range(num_periods + 1):
        ax.axhline(y=p * cell_height, color='black', linewidth=1)

    ts_to_cells: Dict[Tuple[int, int], List[ScheduleAssignment]] = {}
    for a in assignments:
        ts = timeslot_map[a.timeslot_id]
        key = (ts.day, ts.period)
        if key not in ts_to_cells:
            ts_to_cells[key] = []
        ts_to_cells[key].append(a)

    for (d, p), cell_assigns in ts_to_cells.items():
        n_in_cell = len(cell_assigns)
        for idx_in_cell, a in enumerate(cell_assigns):
            course = course_map[a.course_id]
            room = classroom_map[a.classroom_id]

            base_color = color_map[a.course_id]
            is_conflict = (conflicting_courses is not None and
                           a.course_id in conflicting_courses)
            is_adjusted = (adjusted_courses is not None and
                           a.course_id in adjusted_courses)

            sub_width = cell_width / max(n_in_cell, 1)
            x0 = d * cell_width + idx_in_cell * sub_width
            y0 = (num_periods - 1 - p) * cell_height

            rect = Rectangle(
                (x0, y0), sub_width, cell_height,
                facecolor=base_color,
                edgecolor='black',
                linewidth=1.5 if is_conflict else 0.8,
                alpha=0.85
            )
            ax.add_patch(rect)

            if is_adjusted:
                marker = Rectangle(
                    (x0 + 0.02, y0 + cell_height - 0.12), 0.1, 0.1,
                    facecolor='#FFD700',
                    edgecolor='#FF8C00',
                    linewidth=1.5,
                    zorder=10
                )
                ax.add_patch(marker)

            if is_conflict:
                cx = x0 + sub_width / 2
                cy = y0 + cell_height / 2
                ax.plot(cx, cy, marker='X', markersize=16,
                        markeredgecolor='red', markerfacecolor='none',
                        markeredgewidth=2.5, zorder=11)

            lines = [
                f"{course.name}",
                f"教师: {course.teacher}",
                f"班级: {', '.join(course.classes[:2])}",
                f"教室: {room.name}"
            ]
            if len(course.classes) > 2:
                lines[2] += "等"

            font_size = 7 if n_in_cell > 2 else 8
            for li, line in enumerate(lines):
                y_text = y0 + cell_height - 0.18 - li * 0.18
                ax.text(
                    x0 + sub_width / 2, y_text,
                    line,
                    ha='center', va='top',
                    fontsize=font_size,
                    fontweight='bold' if li == 0 else 'normal',
                    color='black',
                    zorder=5
                )

    ax.set_xlim(0, num_days * cell_width)
    ax.set_ylim(0, num_periods * cell_height)

    ax.set_xticks([d * cell_width + cell_width / 2 for d in range(num_days)])
    ax.set_xticklabels(day_names[:num_days], fontsize=12, fontweight='bold')

    ax.set_yticks([p * cell_height + cell_height / 2 for p in range(num_periods)])
    ax.set_yticklabels(period_names[::-1], fontsize=12, fontweight='bold')

    legend_handles = []
    legend_labels = []

    from matplotlib.patches import Patch
    sample_per_course = min(len(expanded_courses), 12)
    for i in range(sample_per_course):
        c = expanded_courses[i]
        legend_handles.append(Patch(facecolor=color_map[c.course_id], alpha=0.85))
        legend_labels.append(f"{c.name}-{c.teacher}")

    if adjusted_courses is not None and len(adjusted_courses) > 0:
        legend_handles.append(Patch(facecolor='#FFD700', edgecolor='#FF8C00', linewidth=1.5))
        legend_labels.append("★ 已调整的课程")

    if conflicting_courses is not None and len(conflicting_courses) > 0:
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], marker='X', color='w', markerfacecolor='none',
                   markeredgecolor='red', markersize=12, markeredgewidth=2,
                   label='冲突标记')
        )
        legend_labels.append("✖ 存在冲突")

    if legend_handles:
        ax.legend(
            legend_handles, legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(5, len(legend_handles)),
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True
        )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"课表图已保存至: {save_path}")
    plt.close()
    return fig


def visualize_convergence(
    history: GAHistory,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
):
    has_diversity = len(history.diversity) > 0
    has_mutation = len(history.mutation_rates) > 0
    nrows = 3 if (has_diversity or has_mutation) else 2

    fig, axes = plt.subplots(nrows, 2, figsize=figsize)
    if nrows == 2:
        axes = axes.reshape(2, 2)

    gens = history.generations

    axes[0, 0].plot(gens, history.best_fitness, color='#2E86AB', linewidth=2, label='最佳适应度')
    axes[0, 0].plot(gens, history.avg_fitness, color='#A23B72', linewidth=1.5, alpha=0.8, label='平均适应度')
    if len(history.worst_fitness) == len(gens):
        axes[0, 0].fill_between(gens, history.worst_fitness, history.best_fitness,
                                alpha=0.15, color='#2E86AB')
    axes[0, 0].set_xlabel('进化代数', fontsize=11)
    axes[0, 0].set_ylabel('适应度', fontsize=11)
    axes[0, 0].set_title('适应度收敛曲线', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    axes[0, 1].plot(gens, history.conflict_counts, color='#C84630', linewidth=2,
                    marker='o', markersize=3, label='冲突数')
    axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                       label='目标线(0)')
    axes[0, 1].set_xlabel('进化代数', fontsize=11)
    axes[0, 1].set_ylabel('冲突数量', fontsize=11)
    axes[0, 1].set_title('冲突数变化趋势', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    if min(history.conflict_counts) >= 0:
        axes[0, 1].set_ylim(bottom=-0.5)

    if len(gens) > 5:
        window = min(10, len(gens) // 5)
        kernel = np.ones(window) / window
        smooth_conflicts = np.convolve(history.conflict_counts, kernel, mode='same')
        axes[0, 1].plot(gens, smooth_conflicts, color='#C84630', linewidth=1.5,
                        linestyle='--', alpha=0.5, label=f'滑动平均(window={window})')
        axes[0, 1].legend(fontsize=10)

    fitness_gain = []
    for i in range(len(history.best_fitness)):
        if i == 0:
            fitness_gain.append(0)
        else:
            gain = history.best_fitness[i] - history.best_fitness[i - 1]
            fitness_gain.append(max(0, gain))

    axes[1, 0].bar(gens, fitness_gain, color='#F18F01', alpha=0.7, width=1.0)
    axes[1, 0].set_xlabel('进化代数', fontsize=11)
    axes[1, 0].set_ylabel('适应度增量', fontsize=11)
    axes[1, 0].set_title('每代适应度改善量', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')

    if has_diversity:
        ax_div = axes[1, 1]
        color_div = '#2A9D8F'
        ax_div.plot(gens, history.diversity, color=color_div, linewidth=2,
                    label='种群多样性')
        ax_div.axhline(y=0.6, color='#8AB17D', linestyle='-.', alpha=0.7,
                       linewidth=1.2, label='多样性充足(0.6)')
        ax_div.axhline(y=0.15, color='#E9C46A', linestyle=':', alpha=0.8,
                       linewidth=1.5, label='变异触发阈值(0.15)')
        if history.restarts > 0:
            ax_div.axhline(y=0.12, color='#E76F51', linestyle='--', alpha=0.7,
                           linewidth=1.2, label='重启阈值(0.12)')
        ax_div.set_xlabel('进化代数', fontsize=11)
        ax_div.set_ylabel('多样性 (汉明距离)', fontsize=11, color=color_div)
        ax_div.tick_params(axis='y', labelcolor=color_div)
        ax_div.set_ylim(bottom=0, top=1.05)
        ax_div.set_title('种群多样性 & 自适应变异率', fontsize=13, fontweight='bold')
        ax_div.grid(True, alpha=0.3, linestyle='--')

        if has_mutation:
            ax_mut = ax_div.twinx()
            color_mut = '#264653'
            ax_mut.plot(gens, history.mutation_rates, color=color_mut,
                        linewidth=1.8, linestyle='-.', alpha=0.85,
                        label='变异概率')
            ax_mut.set_ylabel('变异率', fontsize=11, color=color_mut)
            ax_mut.tick_params(axis='y', labelcolor=color_mut)
            ax_mut.set_ylim(bottom=0, top=0.55)

            lines1, labels1 = ax_div.get_legend_handles_labels()
            lines2, labels2 = ax_mut.get_legend_handles_labels()
            ax_div.legend(lines1 + lines2, labels1 + labels2,
                          fontsize=9, loc='upper right')
        else:
            ax_div.legend(fontsize=9)
    else:
        stats_text = (
            f"总进化代数: {len(gens)}\n"
            f"耗时: {history.elapsed_time:.2f}秒\n"
            f"最终最佳适应度: {history.best_fitness[-1]:.6f}\n"
            f"最终冲突数: {history.conflict_counts[-1]}"
        )
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, stats_text,
                        fontsize=12, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8F9FA',
                                  edgecolor='#DEE2E6', linewidth=2))

    if nrows == 3:
        stats_text = (
            f"总进化代数: {len(gens)}\n"
            f"耗时: {history.elapsed_time:.2f}秒\n"
            f"最终最佳适应度: {history.best_fitness[-1]:.6f}\n"
            f"最终冲突数: {history.conflict_counts[-1]}\n"
            f"触发部分重启: {history.restarts} 次"
        )
        axes[2, 0].axis('off')
        axes[2, 0].text(0.05, 0.95, stats_text,
                        fontsize=11, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8F9FA',
                                  edgecolor='#DEE2E6', linewidth=2))

        if has_diversity and len(gens) > 2:
            diversity_arr = np.array(history.diversity)
            counts, bins = np.histogram(diversity_arr, bins=15, range=(0, 1))
            axes[2, 1].bar(bins[:-1], counts, width=np.diff(bins)[0] * 0.9,
                           color='#457B9D', alpha=0.8, edgecolor='white')
            axes[2, 1].axvline(x=np.mean(diversity_arr), color='#E63946',
                               linestyle='--', linewidth=2,
                               label=f'均值={np.mean(diversity_arr):.3f}')
            axes[2, 1].set_xlabel('多样性区间', fontsize=11)
            axes[2, 1].set_ylabel('代数', fontsize=11)
            axes[2, 1].set_title('多样性分布直方图', fontsize=13, fontweight='bold')
            axes[2, 1].legend(fontsize=10)
            axes[2, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        else:
            axes[2, 1].axis('off')

    title = "遗传算法优化过程可视化"
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"收敛曲线图已保存至: {save_path}")
    plt.close()
    return fig


def visualize_teacher_distribution(
    schedule: Schedule,
    expanded_courses: List[Course],
    assignments: List[ScheduleAssignment],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
):
    timeslot_map = schedule.get_timeslot_map()
    course_map = {c.course_id: c for c in expanded_courses}

    teachers = sorted(set(c.teacher for c in expanded_courses))
    teacher_idx = {t: i for i, t in enumerate(teachers)}

    num_days = 5
    num_periods = 5

    matrix = np.zeros((len(teachers), num_days * num_periods), dtype=np.int32)

    for a in assignments:
        course = course_map[a.course_id]
        ts = timeslot_map[a.timeslot_id]
        ti = teacher_idx[course.teacher]
        si = ts.day * num_periods + ts.period
        matrix[ti, si] = 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect='auto', cmap='Blues', interpolation='nearest',
                   vmin=0, vmax=1)

    ax.set_yticks(range(len(teachers)))
    ax.set_yticklabels(teachers, fontsize=10)

    period_labels = []
    day_names = ["一", "二", "三", "四", "五"]
    for d in range(num_days):
        for p in range(num_periods):
            period_labels.append(f"{day_names[d]}{p+1}")
    ax.set_xticks(range(0, num_days * num_periods, 5))
    ax.set_xticklabels(day_names, fontsize=10)

    for d in range(1, num_days):
        ax.axvline(x=d * num_periods - 0.5, color='black', linewidth=2)

    for i in range(len(teachers)):
        for j in range(num_days * num_periods):
            if matrix[i, j] == 1:
                ax.text(j, i, '课', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')

    ax.set_xlabel('周次 (周)', fontsize=11)
    ax.set_ylabel('教师', fontsize=11)
    ax.set_title('教师课程时间分布热力图', fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1])
    cbar.set_ticklabels(['空闲', '有课'])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"教师分布图已保存至: {save_path}")
    plt.close()
    return fig


def visualize_adjustments(
    schedule: Schedule,
    expanded_courses: List[Course],
    old_assignments: List[ScheduleAssignment],
    new_assignments: List[ScheduleAssignment],
    adjustments: List[Tuple[int, ScheduleAssignment, ScheduleAssignment]],
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
):
    if figsize is None:
        figsize = (14, max(6, len(adjustments) * 0.5 + 2))
    course_map = {c.course_id: c for c in expanded_courses}
    classroom_map = schedule.get_classroom_map()
    timeslot_map = schedule.get_timeslot_map()

    if not adjustments:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "无需调整 - 无冲突!",
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='green',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#D4EDDA',
                          edgecolor='#28A745', linewidth=2))
        ax.axis('off')
        ax.set_title('课程调整报告', fontsize=15, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"调整报告图已保存至: {save_path}")
        plt.close()
        return fig

    n = len(adjustments)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    headers = ['课程名称', '教师', '原安排 (时间/教室)', '新安排 (时间/教室)']
    col_widths = [0.22, 0.12, 0.30, 0.30]
    row_height = 1.0 / (n + 2)
    col_starts = np.cumsum([0] + col_widths)

    for ci, header in enumerate(headers):
        ax.text(
            col_starts[ci] + col_widths[ci] / 2,
            1 - row_height / 2,
            header,
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='#495057',
                      edgecolor='black', linewidth=1)
        )

    for ri, (course_id, old_a, new_a) in enumerate(adjustments):
        y_pos = 1 - (ri + 1.5) * row_height
        course = course_map[course_id]

        old_ts_str = "N/A"
        old_room_str = "N/A"
        if old_a is not None:
            old_ts = timeslot_map[old_a.timeslot_id]
            old_room = classroom_map[old_a.classroom_id]
            old_ts_str = f"{old_ts.day_name} {old_ts.period_name}"
            old_room_str = old_room.name

        new_ts = timeslot_map[new_a.timeslot_id]
        new_room = classroom_map[new_a.classroom_id]
        new_ts_str = f"{new_ts.day_name} {new_ts.period_name}"
        new_room_str = new_room.name

        cells = [
            (course.name, 'black'),
            (course.teacher, 'black'),
            (f"{old_ts_str}\n{old_room_str}", '#C84630'),
            (f"{new_ts_str}\n{new_room_str}", '#2A9D8F'),
        ]

        for ci, (text, color) in enumerate(cells):
            ax.text(
                col_starts[ci] + col_widths[ci] / 2,
                y_pos,
                text,
                ha='center', va='center',
                fontsize=9,
                color=color,
                bbox=dict(boxstyle='square,pad=0.3',
                          facecolor='#F8F9FA' if ri % 2 == 0 else '#E9ECEF',
                          edgecolor='#DEE2E6', linewidth=0.5)
            )

        arrow_x = col_starts[2] + col_widths[2]
        ax.annotate(
            '', xy=(col_starts[3] + 0.005, y_pos),
            xytext=(arrow_x - 0.005, y_pos),
            arrowprops=dict(arrowstyle='->', color='#495057', lw=1.5)
        )

    title = f'课程调整报告 (共调整 {n} 门课程)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"调整报告图已保存至: {save_path}")
    plt.close()
    return fig


def visualize_conflict_detailed(
    report: ConflictReport,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
):
    categories = ['教师冲突', '班级冲突', '教室冲突', '容量冲突', '类型冲突']
    values = [
        report.teacher_conflicts,
        report.class_conflicts,
        report.classroom_conflicts,
        report.capacity_violations,
        report.type_violations,
    ]
    colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#457B9D']

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    bars = axes[0].bar(categories, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    axes[0].set_xlabel('冲突类型', fontsize=11)
    axes[0].set_ylabel('冲突数量', fontsize=11)
    axes[0].set_title('各类冲突数量统计', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     str(int(val)), ha='center', va='bottom',
                     fontsize=11, fontweight='bold')

    non_zero = [(c, v, col) for c, v, col in zip(categories, values, colors) if v > 0]
    if non_zero:
        labels, sizes, pie_colors = zip(*non_zero)
        axes[1].pie(sizes, labels=labels, colors=pie_colors,
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops=dict(edgecolor='white', linewidth=2))
        axes[1].set_title('冲突类型占比', fontsize=13, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, '无冲突!', ha='center', va='center',
                     fontsize=20, fontweight='bold', color='green',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#D4EDDA',
                               edgecolor='#28A745', linewidth=2))
        axes[1].axis('off')

    summary = f'总冲突数: {report.total_conflicts()}'
    fig.suptitle(summary, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"冲突分析图已保存至: {save_path}")
    plt.close()
    return fig
