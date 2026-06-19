import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import time

from data_structures import (
    Course, Classroom, TimeSlot, Schedule, ScheduleAssignment,
    expand_courses_to_sessions
)
from conflict_detector import (
    ConflictDetector, ConflictReport, Conflict, generate_random_schedule
)


@dataclass
class GAHistory:
    generations: List[int] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    avg_fitness: List[float] = field(default_factory=list)
    worst_fitness: List[float] = field(default_factory=list)
    conflict_counts: List[int] = field(default_factory=list)
    diversity: List[float] = field(default_factory=list)
    mutation_rates: List[float] = field(default_factory=list)
    restarts: int = 0
    elapsed_time: float = 0.0


class GeneticAlgorithmScheduler:
    def __init__(
        self,
        schedule: Schedule,
        expanded_courses: List[Course],
        population_size: int = 100,
        max_generations: int = 500,
        mutation_rate: float = 0.08,
        crossover_rate: float = 0.90,
        elite_count: int = 5,
        tournament_size: int = 5,
        random_seed: Optional[int] = None,
        diversity_preserve: bool = True,
        adaptive_mutation: bool = True,
        migration_rate: float = 0.05,
        fitness_sharing: bool = True,
        share_radius: float = 0.15,
        restart_threshold: float = 0.12,
        restart_fraction: float = 0.35,
    ):
        self.schedule = schedule
        self.expanded_courses = expanded_courses
        self.population_size = population_size
        self.max_generations = max_generations
        self.base_mutation_rate = mutation_rate
        self.current_mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size

        self.diversity_preserve = diversity_preserve
        self.adaptive_mutation = adaptive_mutation
        self.migration_rate = migration_rate
        self.fitness_sharing = fitness_sharing
        self.share_radius = share_radius
        self.restart_threshold = restart_threshold
        self.restart_fraction = restart_fraction

        if random_seed is not None:
            np.random.seed(random_seed)

        self.num_courses = len(expanded_courses)
        self.num_timeslots = schedule.num_timeslots()
        self.num_classrooms = schedule.num_classrooms()

        self.temp_schedule = Schedule(
            courses=expanded_courses,
            classrooms=schedule.classrooms,
            timeslots=schedule.timeslots,
            assignments=[]
        )
        self.detector = ConflictDetector(self.temp_schedule)
        self.course_map = self.temp_schedule.get_course_map()
        self.classroom_map = schedule.get_classroom_map()

        self.valid_rooms_per_course = self._compute_valid_rooms()
        self.valid_rooms_array = self._valid_rooms_to_array()
        self.history = GAHistory()

        self._build_feature_tensors()

    def _compute_valid_rooms(self) -> Dict[int, List[int]]:
        valid_rooms = {}
        for i, course in enumerate(self.expanded_courses):
            valid = [
                r.classroom_id for r in self.schedule.classrooms
                if r.capacity >= course.student_count
            ]
            if not valid:
                valid = [r.classroom_id for r in self.schedule.classrooms]
            valid_rooms[i] = valid
        return valid_rooms

    def _valid_rooms_to_array(self) -> np.ndarray:
        max_rooms = max(len(v) for v in self.valid_rooms_per_course.values())
        arr = np.zeros((self.num_courses, max_rooms), dtype=np.int32)
        counts = np.zeros(self.num_courses, dtype=np.int32)
        for i in range(self.num_courses):
            rooms = self.valid_rooms_per_course[i]
            counts[i] = len(rooms)
            arr[i, :len(rooms)] = rooms
        self.valid_room_counts = counts
        return arr

    def _build_feature_tensors(self):
        teachers = sorted(set(c.teacher for c in self.expanded_courses))
        self.teacher_list = teachers
        self.num_teachers = len(teachers)
        self.teacher_ids = np.array(
            [teachers.index(c.teacher) for c in self.expanded_courses],
            dtype=np.int32
        )

        all_classes = sorted(set(
            cls for c in self.expanded_courses for cls in c.classes
        ))
        self.class_list = all_classes
        self.num_all_classes = len(all_classes)
        self.class_membership = np.zeros(
            (self.num_courses, self.num_all_classes), dtype=np.int8
        )
        for i, c in enumerate(self.expanded_courses):
            for cls in c.classes:
                self.class_membership[i, all_classes.index(cls)] = 1

        self.capacities = np.array(
            [r.capacity for r in self.schedule.classrooms], dtype=np.int32
        )
        self.student_counts = np.array(
            [c.student_count for c in self.expanded_courses], dtype=np.int32
        )

    def _evaluate_fitness_vectorized(
        self, population: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        P = population.shape[0]
        N = self.num_courses
        T = self.num_timeslots
        R = self.num_classrooms

        ts = population[:, :, 0]
        rm = population[:, :, 1]

        ts_onehot = np.zeros((P, N, T), dtype=np.int8)
        ts_onehot[np.arange(P)[:, None], np.arange(N)[None, :], ts] = 1

        teacher_conflicts = np.zeros(P, dtype=np.float64)
        for tid in range(self.num_teachers):
            mask = self.teacher_ids == tid
            if mask.sum() < 2:
                continue
            count = ts_onehot[:, mask, :].sum(axis=1)
            teacher_conflicts += np.maximum(count - 1, 0).sum(axis=1)

        class_conflicts = np.zeros(P, dtype=np.float64)
        for cid in range(self.num_all_classes):
            mask = self.class_membership[:, cid] == 1
            if mask.sum() < 2:
                continue
            count = ts_onehot[:, mask, :].sum(axis=1)
            class_conflicts += np.maximum(count - 1, 0).sum(axis=1)

        rm_onehot = np.zeros((P, N, R), dtype=np.int8)
        rm_onehot[np.arange(P)[:, None], np.arange(N)[None, :], rm] = 1

        ts_expanded = ts[:, :, None, None]  # (P, N, 1, 1)
        ts_match = (ts_expanded == ts[:, None, :, None])  # (P, N, N, 1)
        rm_expanded = rm[:, :, None, None]  # (P, N, 1, 1)
        rm_match = (rm_expanded == rm[:, None, :, None])  # (P, N, N, 1)
        same_slot_room = (ts_match & rm_match).astype(np.int8)[:, :, :, 0]  # (P, N, N)
        diag_mask = ~np.eye(N, dtype=bool)[None, :, :]
        classroom_conflicts = (same_slot_room * diag_mask).sum(axis=(1, 2)) // 2
        classroom_conflicts = classroom_conflicts.astype(np.float64)

        student_counts_3d = self.student_counts[None, :, None]
        capacities_3d = self.capacities[None, None, :]
        overload_mask = student_counts_3d > capacities_3d
        room_usage = (student_counts_3d * overload_mask.astype(np.int32)
                      ) * rm_onehot
        capacity_violations = (room_usage > 0).any(axis=1).sum(axis=1)
        capacity_violations = capacity_violations.astype(np.float64)

        type_violations = np.zeros(P, dtype=np.float64)

        ts_usage = ts_onehot.sum(axis=1)
        variance_penalty = np.var(ts_usage, axis=1) * 0.1

        total_penalty = (
            teacher_conflicts * 10.0
            + class_conflicts * 10.0
            + classroom_conflicts * 10.0
            + capacity_violations * 8.0
            + type_violations * 5.0
            + variance_penalty
        )

        fitnesses = 1.0 / (1.0 + total_penalty)

        total_conflicts = (
            teacher_conflicts.astype(np.int32)
            + class_conflicts.astype(np.int32)
            + classroom_conflicts.astype(np.int32)
            + capacity_violations.astype(np.int32)
        )

        return (
            fitnesses,
            teacher_conflicts,
            class_conflicts,
            classroom_conflicts,
            total_conflicts,
        )

    def _generate_report_from_tensors(
        self,
        chromosome: np.ndarray,
        teacher_c: int,
        class_c: int,
        room_c: int,
        cap_c: int,
    ) -> ConflictReport:
        assignments = self._decode_chromosome(chromosome)
        report = self.detector.detect(assignments)
        return report

    def _compute_population_diversity(self, population: np.ndarray) -> float:
        P = population.shape[0]
        if P <= 1:
            return 1.0

        sample_size = min(P, 30)
        indices = np.random.choice(P, sample_size, replace=False)
        sample = population[indices]

        ts = sample[:, :, 0].astype(np.int64)
        rm = sample[:, :, 1].astype(np.int64)

        combined = ts * (self.num_classrooms + 1) + rm

        distances = []
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                diff = (combined[i] != combined[j]).sum()
                distances.append(diff / self.num_courses)

        if not distances:
            return 1.0
        return float(np.mean(distances))

    def _apply_fitness_sharing(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        if not self.fitness_sharing:
            return fitnesses

        P = population.shape[0]
        ts = population[:, :, 0].astype(np.int64)
        rm = population[:, :, 1].astype(np.int64)
        combined = ts * (self.num_classrooms + 1) + rm

        shared = fitnesses.copy()
        sample_p = min(P, 100)
        if sample_p < P:
            sample_idx = np.random.choice(P, sample_p, replace=False)
            combined_sub = combined[sample_idx]
            fit_sub = fitnesses[sample_idx]
        else:
            combined_sub = combined
            fit_sub = fitnesses
            sample_idx = np.arange(P)

        for i in range(P):
            diff = (combined[i] != combined_sub[:, None]).sum(axis=1) / self.num_courses
            sh = np.maximum(0.0, 1.0 - (diff / self.share_radius) ** 2)
            niche_count = sh.sum()
            if niche_count > 1.0:
                shared[i] = fitnesses[i] / (niche_count ** 0.5)

        return shared

    def _decode_chromosome(self, chromosome: np.ndarray) -> List[ScheduleAssignment]:
        assignments = []
        for i in range(self.num_courses):
            ts_id = int(chromosome[i, 0])
            room_id = int(chromosome[i, 1])
            assignments.append(ScheduleAssignment(
                course_id=i,
                timeslot_id=ts_id,
                classroom_id=room_id
            ))
        return assignments

    def _encode_chromosome(self, assignments: List[ScheduleAssignment]) -> np.ndarray:
        chrom = np.zeros((self.num_courses, 2), dtype=np.int32)
        for a in assignments:
            chrom[a.course_id, 0] = a.timeslot_id
            chrom[a.course_id, 1] = a.classroom_id
        return chrom

    def _random_chromosome(self) -> np.ndarray:
        chrom = np.zeros((self.num_courses, 2), dtype=np.int32)
        chrom[:, 0] = np.random.randint(0, self.num_timeslots, size=self.num_courses)
        room_indices = (
            np.random.random(self.num_courses) * self.valid_room_counts
        ).astype(np.int32)
        chrom[:, 1] = self.valid_rooms_array[
            np.arange(self.num_courses), room_indices
        ]
        return chrom

    def _initialize_population(self, initial_assignments: List[ScheduleAssignment]) -> np.ndarray:
        population = np.zeros(
            (self.population_size, self.num_courses, 2), dtype=np.int32
        )

        start_idx = 0
        if initial_assignments:
            population[0] = self._encode_chromosome(initial_assignments)
            start_idx = 1
            for p in range(1, min(self.population_size // 5, 20)):
                chrom = population[0].copy()
                n_mutate = max(1, int(self.num_courses * 0.15 * p / 20))
                mutate_idx = np.random.choice(
                    self.num_courses, n_mutate, replace=False
                )
                for idx in mutate_idx:
                    if np.random.random() < 0.5:
                        chrom[idx, 0] = np.random.randint(0, self.num_timeslots)
                    else:
                        rooms = self.valid_rooms_per_course[idx]
                        chrom[idx, 1] = rooms[np.random.randint(0, len(rooms))]
                population[p] = chrom
            start_idx = min(self.population_size // 5, 20)

        for p in range(start_idx, self.population_size):
            population[p] = self._random_chromosome()

        return population

    def _tournament_selection(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> np.ndarray:
        idx = np.random.choice(
            self.population_size, self.tournament_size, replace=False
        )
        best_idx = idx[np.argmax(fitnesses[idx])]
        return population[best_idx].copy()

    def _rank_selection_probs(self, fitnesses: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(fitnesses)) + 1
        probs = ranks / ranks.sum()
        return probs

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        method = np.random.random()
        if method < 0.4:
            point = np.random.randint(1, self.num_courses)
            child1[point:] = parent2[point:]
            child2[point:] = parent1[point:]
        elif method < 0.7:
            p1 = np.random.randint(0, self.num_courses)
            p2 = np.random.randint(p1 + 1, self.num_courses + 1)
            child1[p1:p2] = parent2[p1:p2]
            child2[p1:p2] = parent1[p1:p2]
        else:
            mask = np.random.random(self.num_courses) < 0.5
            swap_mask = np.stack([mask, mask], axis=1)
            child1 = np.where(swap_mask, parent2, parent1)
            child2 = np.where(swap_mask, parent1, parent2)

        return child1, child2

    def _mutate_vectorized(
        self,
        population_batch: np.ndarray,
        conflict_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        B = population_batch.shape[0]
        mutated = population_batch.copy()

        base_rate = self.current_mutation_rate

        mutation_probs = np.full(
            (B, self.num_courses), base_rate, dtype=np.float64
        )

        if conflict_mask is not None:
            boost = 0.3
            mutation_probs = np.where(
                conflict_mask[:, :, None].any(axis=2)
                if conflict_mask.ndim == 3 else conflict_mask,
                np.maximum(mutation_probs, boost),
                mutation_probs,
            )

        mutate_mask = np.random.random((B, self.num_courses)) < mutation_probs
        which_gene = np.random.random((B, self.num_courses)) < 0.5

        ts_mask = mutate_mask & which_gene
        if ts_mask.any():
            n_ts = ts_mask.sum()
            mutated[ts_mask, 0] = np.random.randint(
                0, self.num_timeslots, size=n_ts
            )

        rm_mask = mutate_mask & (~which_gene)
        if rm_mask.any():
            batch_idx, course_idx = np.where(rm_mask)
            n_rm = len(batch_idx)
            counts = self.valid_room_counts[course_idx]
            room_positions = (np.random.random(n_rm) * counts).astype(np.int32)
            mutated[batch_idx, course_idx, 1] = self.valid_rooms_array[
                course_idx, room_positions
            ]

        return mutated

    def _inject_migrants(self, population: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return population
        indices = np.random.choice(
            self.population_size, count, replace=False
        )
        for idx in indices:
            population[idx] = self._random_chromosome()
        return population

    def _partial_restart(
        self,
        population: np.ndarray,
        fitnesses: np.ndarray,
        best_chromosome: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        n_restart = int(self.population_size * self.restart_fraction)
        worst_indices = np.argsort(fitnesses)[:n_restart]

        restarted = 0
        for idx in worst_indices:
            if np.random.random() < 0.5:
                population[idx] = self._random_chromosome()
            else:
                chrom = best_chromosome.copy()
                n_shake = max(1, int(self.num_courses * 0.25))
                shake_idx = np.random.choice(
                    self.num_courses, n_shake, replace=False
                )
                for si in shake_idx:
                    if np.random.random() < 0.5:
                        chrom[si, 0] = np.random.randint(0, self.num_timeslots)
                    else:
                        rooms = self.valid_rooms_per_course[si]
                        chrom[si, 1] = rooms[np.random.randint(0, len(rooms))]
                population[idx] = chrom
            restarted += 1

        return population, restarted

    def _update_adaptive_mutation(
        self, diversity: float, no_improvement: int
    ):
        if not self.adaptive_mutation:
            return

        base = self.base_mutation_rate
        target_div_low = 0.15
        target_div_high = 0.6

        div_factor = 1.0
        if diversity < target_div_low:
            div_factor = 1.0 + (target_div_low - diversity) * 15.0
        elif diversity > target_div_high:
            div_factor = max(0.3, 1.0 - (diversity - target_div_high) * 2.0)

        stagnation_factor = 1.0 + min(no_improvement / 80.0, 4.0)

        new_rate = min(max(base * div_factor * stagnation_factor, base * 0.3), 0.45)
        self.current_mutation_rate = new_rate

    def _evaluate_fitness(self, chromosome: np.ndarray) -> Tuple[float, ConflictReport]:
        pop = chromosome[None, :, :]
        (f_arr, t_c, c_c, r_c, tc_c) = self._evaluate_fitness_vectorized(pop)
        report = self._generate_report_from_tensors(
            chromosome, int(t_c[0]), int(c_c[0]), int(r_c[0]), 0
        )
        return float(f_arr[0]), report

    def _mutate(
        self, chromosome: np.ndarray, report: Optional[ConflictReport] = None
    ) -> np.ndarray:
        batch = chromosome[None, :, :]
        mutated_batch = self._mutate_vectorized(batch)
        return mutated_batch[0]

    def optimize(
        self,
        initial_assignments: List[ScheduleAssignment],
        target_conflicts: int = 0,
        patience: int = 150,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[List[ScheduleAssignment], ConflictReport, GAHistory]:
        start_time = time.time()

        population = self._initialize_population(initial_assignments)
        (
            fitnesses,
            t_conf,
            c_conf,
            r_conf,
            total_conf,
        ) = self._evaluate_fitness_vectorized(population)
        shared_fitnesses = self._apply_fitness_sharing(population, fitnesses)

        best_idx = np.argmax(fitnesses)
        best_chromosome = population[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])
        best_total_conf = int(total_conf[best_idx])
        best_report = self._generate_report_from_tensors(
            best_chromosome,
            int(t_conf[best_idx]), int(c_conf[best_idx]),
            int(r_conf[best_idx]), 0,
        )

        no_improvement_count = 0
        total_restarts = 0

        for gen in range(self.max_generations):
            diversity = self._compute_population_diversity(population)
            self._update_adaptive_mutation(diversity, no_improvement_count)

            sorted_idx = np.argsort(shared_fitnesses)[::-1]
            new_population = np.zeros_like(population)

            for i in range(self.elite_count):
                new_population[i] = population[sorted_idx[i]].copy()

            new_idx = self.elite_count
            selection_fitnesses = shared_fitnesses

            conflict_info_per_chrom = None

            while new_idx < self.population_size:
                p1 = self._tournament_selection(population, selection_fitnesses)
                p2 = self._tournament_selection(population, selection_fitnesses)
                c1, c2 = self._crossover(p1, p2)

                children_batch = np.stack([c1, c2], axis=0)
                (f_c, t_c, cc_c, rc_c, tc_c) = self._evaluate_fitness_vectorized(
                    children_batch
                )
                child_conflict_mask = None

                mutated_batch = self._mutate_vectorized(
                    children_batch, child_conflict_mask
                )

                if new_idx < self.population_size:
                    new_population[new_idx] = mutated_batch[0]
                    new_idx += 1
                if new_idx < self.population_size:
                    new_population[new_idx] = mutated_batch[1]
                    new_idx += 1

            n_migrants = int(self.population_size * self.migration_rate)
            if n_migrants > 0 and np.random.random() < 0.5:
                migrant_indices = np.random.choice(
                    range(self.elite_count, self.population_size),
                    n_migrants, replace=False
                )
                for midx in migrant_indices:
                    new_population[midx] = self._random_chromosome()

            population = new_population
            (
                fitnesses,
                t_conf, c_conf, r_conf, total_conf,
            ) = self._evaluate_fitness_vectorized(population)
            shared_fitnesses = self._apply_fitness_sharing(population, fitnesses)

            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = float(fitnesses[current_best_idx])
            current_conf = int(total_conf[current_best_idx])

            improved = False
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[current_best_idx].copy()
                best_total_conf = current_conf
                best_report = self._generate_report_from_tensors(
                    best_chromosome,
                    int(t_conf[current_best_idx]),
                    int(c_conf[current_best_idx]),
                    int(r_conf[current_best_idx]), 0,
                )
                no_improvement_count = 0
                improved = True
            else:
                no_improvement_count += 1

            if (not improved and self.diversity_preserve and
                    diversity < self.restart_threshold and
                    no_improvement_count >= 30):
                population, restarts = self._partial_restart(
                    population, fitnesses, best_chromosome
                )
                total_restarts += restarts
                (
                    fitnesses,
                    t_conf, c_conf, r_conf, total_conf,
                ) = self._evaluate_fitness_vectorized(population)
                shared_fitnesses = self._apply_fitness_sharing(
                    population, fitnesses
                )
                no_improvement_count = 0
                if verbose:
                    print(f"  ---> 第{gen}代触发部分重启 (多样性={diversity:.3f})")

            avg_fit = float(np.mean(fitnesses))
            worst_fit = float(np.min(fitnesses))
            conflict_count = best_total_conf

            self.history.generations.append(gen)
            self.history.best_fitness.append(best_fitness)
            self.history.avg_fitness.append(avg_fit)
            self.history.worst_fitness.append(worst_fit)
            self.history.conflict_counts.append(conflict_count)
            self.history.diversity.append(diversity)
            self.history.mutation_rates.append(self.current_mutation_rate)

            if verbose and (gen % 10 == 0 or gen == self.max_generations - 1 or improved):
                mark = " *" if improved else ""
                print(f"第{gen:4d}代 | 最佳: {best_fitness:.6f} | "
                      f"平均: {avg_fit:.6f} | 冲突: {conflict_count:3d} | "
                      f"多样性: {diversity:.3f} | 变异率: {self.current_mutation_rate:.3f}"
                      f"{mark}")

            if progress_callback:
                progress_callback(gen, best_fitness, avg_fit, conflict_count)

            if conflict_count <= target_conflicts:
                if verbose:
                    print(f"\n达到目标! 第{gen}代冲突数为{conflict_count}")
                break

            if no_improvement_count >= patience:
                if verbose:
                    print(f"\n连续{patience}代无改善，提前终止")
                break

        self.history.elapsed_time = time.time() - start_time
        self.history.restarts = total_restarts

        best_assignments = self._decode_chromosome(best_chromosome)
        final_report = self.detector.detect(best_assignments)

        return best_assignments, final_report, self.history


def find_adjustments(
    old_assignments: List[ScheduleAssignment],
    new_assignments: List[ScheduleAssignment]
) -> List[Tuple[int, ScheduleAssignment, ScheduleAssignment]]:
    old_map = {a.course_id: a for a in old_assignments}
    adjustments = []
    for new_a in new_assignments:
        old_a = old_map.get(new_a.course_id)
        if old_a is None:
            adjustments.append((new_a.course_id, None, new_a))
        elif (old_a.timeslot_id != new_a.timeslot_id or
              old_a.classroom_id != new_a.classroom_id):
            adjustments.append((new_a.course_id, old_a, new_a))
    return adjustments
