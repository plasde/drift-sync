

def greedy_local_optimization(start, goal, wind_field,
                        step_size=1.0,
                        angle_resolution=15,
                        tacking_penalty=3):
    """
    Compute a sailing path that minimizes travel time using polar performance.
    A tacking penalty is added when switching tack sides.
    Returns a list of np.array waypoints.
    """
    start = np.array(start, dtype=np.float64)
    goal = np.array(goal, dtype=np.float64)
    current = start.copy()
    path = [current.copy()]

    max_steps = 1000
    prev_tack_side = None

    for _ in range(max_steps):
        if np.linalg.norm(goal - current) < step_size:
            break

        x = int(np.clip(current[0], 0, wind_field.width - 1))
        y = int(np.clip(current[1], 0, wind_field.height - 1))
        wind_vec = wind_field.get_vector(x, y)

        wind_angle = atan2(wind_vec[1], wind_vec[0])
        vec_to_goal = goal - current

        best_score = -np.inf
        best_heading = None

        for angle_offset in range(-90, 91, angle_resolution):
            heading = atan2(vec_to_goal[1], vec_to_goal[0]) + radians(angle_offset)
            apparent_angle = (heading - wind_angle + pi) % (2 * pi) - pi

            speed = polar_performance(apparent_angle)
            dx = cos(heading) * speed
            dy = sin(heading) * speed
            # normalize for unit projection
            progress = np.dot([dx, dy], vec_to_goal / (np.linalg.norm(vec_to_goal) + 1e-8))

            # Determine tack side
            tack_side = np.sign(cos(heading - wind_angle))
            if prev_tack_side is not None and tack_side != prev_tack_side:
                progress -= tacking_penalty  # apply penalty on tack switch

            if progress > best_score:
                best_score = progress
                best_heading = heading
                best_tack_side = tack_side

        if best_heading is None:
            break  # deadlock

        dx = cos(best_heading) * step_size
        dy = sin(best_heading) * step_size
        current = current + np.array([dx, dy])
        path.append(current.copy())
        prev_tack_side = best_tack_side

    path.append(goal)
    return path


    # def heuristic(a, b):
    #     def estimate_minimum_tacks(a, b , wind_dir, min_tack_distance = 40.0, closehauled_angle = 60.0):
    #         """Estimate minimum number of tacks needed to reach goal."""
    #         # Vector calculations
    #         to_goal = np.array(b) - np.array(a)
    #         distance = np.linalg.norm(to_goal)
    #         goal_dir = to_goal / (distance + 1e-6)
    
    #         # Calculate angle between wind and goal
    #         wind_goal_angle = normalize_angle(
    #             atan2(wind_dir[1], wind_dir[0]) - atan2(goal_dir[1], goal_dir[0]))
    
    #         # If goal is upwind (within no-go zone)
    #         if np.degrees(wind_goal_angle) <= 30:
    #             return 2
            
    #         # Calculate direct upwind component of the distance
    #         #upwind_component = distance * np.cos(wind_goal_angle)
        
    #         # Assuming minimum tacking angle of 45° from wind
    #         #upwind_gain_per_tack = min_tack_distance * np.cos(radians(closehauled_angle))
    #         #num_tacks = max(1, int(abs(upwind_component / upwind_gain_per_tack)))

    #         num_tacks = 1
    #         return num_tacks

    
    #     # Use directional heuristic weighted by goal alignment
    #     to_goal = np.array(b) - np.array(a)
    #     distance = np.linalg.norm(to_goal)

    #     # Clip to grid
    #     x = int(np.clip(a[0], 0, wind_field.width - 1))
    #     y = int(np.clip(a[1], 0, wind_field.height - 1))
    #     wind_vec = np.array(wind_field.get_vector(x, y))

    #     wind_speed = np.linalg.norm(wind_vec)
    #     wind_dir = wind_vec / (wind_speed + 1e-6)

    #     goal_dir = to_goal / (distance + 1e-6)
    #     wind_angle_to_goal = normalize_angle(
    #         atan2(wind_dir[1], wind_dir[0]) - atan2(goal_dir[1], goal_dir[0]))
        
    #     if degrees(wind_angle_to_goal) < 30:
    #         closehauled_angle = 45
    #         perf = polar_performance(radians(closehauled_angle))
    #         estimated_vmg = perf * wind_speed * cos(radians(closehauled_angle))
    #     else:                        
    #         estimated_vmg = polar_performance(wind_angle_to_goal) * wind_speed

    #     time_estimate = distance / (estimated_vmg + 1e-6)  # Avoid division by zero       

    #     min_tacks = estimate_minimum_tacks(a, b, wind_dir)
    #     min_tack_cost = min_tacks * 10.0

    #     print(f"Heuristic: dist={distance:.1f}, wind_angle={degrees(wind_angle_to_goal):.1f}°, vmg={estimated_vmg:.6f}, time={time_estimate:.2f}")
        #return time_estimate + min_tack_cost




# Trajectory optimization sailing pathfinder using VMG-based cost with final speed cap and tack constraints
def trajectory_optimized_path(start, goal, wind_field, num_steps=100, dt=1.0, max_final_speed=0.2, max_final_distance=1.0, min_tack_speed=0.05):
    """
    Compute an optimal sailing path from start to goal using trajectory optimization.
    Uses VMG (Velocity Made Good) as cost signal, penalizes high final speed,
    and only allows tack switches if speed exceeds min_tack_speed.
    """
    start = np.array(start)
    goal = np.array(goal)

    BAD_PENALTY = 1e4

    def simulate_trajectory(heading_deltas):
        pos = start.copy()
        wind_vec = np.array(wind_field.get_vector(int(pos[0]), int(pos[1])))
        wind_angle = atan2(wind_vec[1], wind_vec[0])
        to_goal = goal - pos
        heading = atan2(to_goal[1], to_goal[0]) #wind_angle + radians(45)
        trajectory = [pos.copy()]
        total_cost = 0.0

        tack_count = 0
        prev_tack_side = None
        min_speed_seen = float('inf')

        for i, delta in enumerate(heading_deltas):
            heading += delta
            x = int(np.clip(pos[0], 0, wind_field.width - 1))
            y = int(np.clip(pos[1], 0, wind_field.height - 1))
            wind_vec = np.array(wind_field.get_vector(x, y))
            wind_angle = atan2(wind_vec[1], wind_vec[0])

            apparent_angle = (heading - wind_angle + pi) % (2 * pi) - pi
            #speed = polar_performance(apparent_angle)

            wind_mag = np.linalg.norm(wind_vec)
            speed = polar_performance(apparent_angle) * wind_mag

            progress_ratio = i / num_steps
            speed_weight = max(0.0, 1.0 - (progress_ratio / 0.1))
            vmg_weight = 1.0 - speed_weight

            if speed <= 0.0:
                total_cost += BAD_PENALTY
                speed = 0.02
            
            min_speed_seen = min(min_speed_seen, speed)
            
            if i == 0:
                print(f"Initial heading: {degrees(heading):.1f}, wind angle: {degrees(wind_angle):.1f}, apparent angle: {degrees(apparent_angle):.1f}, polar speed: {speed:.3f}")
            if i % 5 == 0:
                print(f"Apparent angle: {degrees(apparent_angle):.1f}, polar speed: {speed:.3f}")

            to_goal = goal - pos
            goal_angle = atan2(to_goal[1], to_goal[0])

            # Tack side detection
            tack_side = np.sign(cos(heading - wind_angle))
            if prev_tack_side is not None and tack_side != prev_tack_side:
                if speed >= min_tack_speed:
                    tack_count += 1
                    print(f"  Tack switch detected. Speed: {speed:.3f} (Legal Tack ✔️)")
                else:
                    total_cost += 500.0  # large penalty for illegal tack
                    print(f"  Tack switch detected. Speed: {speed:.3f} (Illegal Tack ❌) Penalty applied: +500")
            prev_tack_side = tack_side

            # Testing
            #print(f"tack side: {tack_side}, speed: {speed:.3f}")

            # VMG-based cost
            vmg = speed * cos(heading - goal_angle)

            print(f"Step {i}: VMG: {vmg:.4f}, Heading: {np.degrees(heading):.1f}, Goal angle: {np.degrees(goal_angle):.1f}")
            bad_vmg_counter = 0
            if speed_weight > 0:
                total_cost += 2.0 / (speed + 1e-3) * speed_weight  # reward moving fast
            if vmg_weight > 0:
                if vmg < -0.005:
                    bad_vmg_counter += 1
                    if bad_vmg_counter >= 3:
                        total_cost += 1e6  # Catastrophic penalty
                        print("Emergency: Persistent Negative VMG! Heavy penalty applied.")
                
                    total_cost += 10000.0 * vmg_weight
                elif vmg < 0:
                    total_cost += 10000.0 * abs(vmg) * vmg_weight
                    bad_vmg_counter = 0
                else:
                    total_cost += 5.0 / (vmg + 1e-3) * vmg_weight
                    bad_vmg_counter = 0

            print(f"Total cost is now: {total_cost:.4f}")

            # Encourage distance closing
            distance_to_goal = np.linalg.norm(to_goal)
            dist_penalty = 0.05 * distance_to_goal
            total_cost += dist_penalty

            # Apply motion
            dx = speed * cos(heading) * dt
            dy = speed * sin(heading) * dt
            
            pos += np.array([dx, dy])
            trajectory.append(pos.copy())

            print(f"Step {i}: Heading: {degrees(heading):.1f}°, Goal: {degrees(goal_angle):.1f}°, Delta: {degrees(delta):.1f}°")
            print(f"  Speed: {speed:.3f}, VMG: {vmg:.4f}, Distance to Goal: {distance_to_goal:.2f}")
            print(f"  Total cost so far: {total_cost:.2f}")
            print("-------------------------------------")        

        # Final speed check
        # Recalculate final wind and apparent angle at final pos
        #x = int(np.clip(pos[0], 0, wind_field.width - 1))
        #y = int(np.clip(pos[1], 0, wind_field.height - 1))

        #wind_vec = np.array(wind_field.get_vector(x, y))
        #wind_angle = atan2(wind_vec[1], wind_vec[0])
        #apparent_angle = (heading - wind_angle + pi) % (2 * pi) - pi

        #final_speed = polar_performance(apparent_angle)
        #if final_speed > max_final_speed:
        #    total_cost += (final_speed - max_final_speed) ** 2
        #if np.isnan(total_cost) or np.isinf(total_cost):
        #    total_cost = BAD_PENALTY

        final_distance = np.linalg.norm(goal - pos)

        return total_cost, trajectory, final_distance, min_speed_seen

    def objective(heading_deltas):
        cost, _, _, _ = simulate_trajectory(heading_deltas)
        # Small heading change penalty
        smoothness_penalty = 0.1 * np.sum(np.square(np.diff(heading_deltas)))
        print(f"Smoothness penalty: {smoothness_penalty:.4f}")
        return cost + smoothness_penalty
    
    def constraint_final_position(heading_deltas):
        _, _, final_distance, _ = simulate_trajectory(heading_deltas)
        return max_final_distance - final_distance
    
    def constraint_tack_speed(heading_deltas):
        _, _, _, min_speed = simulate_trajectory(heading_deltas)
        return min_speed - 0.01

    initial_deltas = np.random.uniform(-radians(20), radians(20), num_steps)
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_final_position},
        {'type': 'ineq', 'fun': constraint_tack_speed}
    ]

    result = minimize(
                    objective,
                    initial_deltas,
                    method='COBYLA',
                    constraints=constraints,
                    options={
                        'maxiter': 1000,
                        'rhobeg': 0.1,
                        'disp': True
                    })

    _, best_trajectory, _, _ = simulate_trajectory(result.x)
    return best_trajectory