from vehicle import Driver
from controller import Lidar, InertialUnit, Gyro, Accelerometer
import math
import numpy as np

driver = Driver()
basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

#capteurs
lidar = Lidar("RpLidarA2")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud() 
imu = InertialUnit("imu")
imu.enable(sensorTimeStep)
gyro = Gyro("gyro")
gyro.enable(sensorTimeStep)
accelerometer = Accelerometer("accelerometer")
accelerometer.enable(sensorTimeStep)
keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

#Parametres
speed, angle = 0, 0
maxSpeed, maxangle = 28, 0.3
obstacle_threshold, safe_distance, collision_threshold = 2.5, 3.25, 1.5
last_angle = 0
angle_history = []
emergency_mode, emergency_counter, turn_detected = False, 0, False

uturn_mode, uturn_state, uturn_direction, uturn_timer = False, 0, 0, 0
uturn_initial_yaw, uturn_target_yaw = 0.0, 0.0

driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)
modeManuel, modeAuto = False, False
print_counter, PRINT_INTERVAL = 0, 5

MAP_SIZE, MAP_RESOLUTION, MAP_ORIGIN = 200, 0.05, 100
occupancy_grid = np.ones((MAP_SIZE, MAP_SIZE)) * 0.5
robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
last_yaw = None
trajectory_history = []
slam_update_counter, SLAM_UPDATE_INTERVAL = 0, 3
curve_detected_ahead, loop_closure_detected = False, False
CONTRESENS_HISTORY_MIN, SHARP_TURN_YAW_RATE = 30, 0.5
contresens_counter, CONTRESENS_CONFIRM_FRAMES = 0, 8

#fonction nécessaires
def get_min_distance(donnees_lidar, start_angle, end_angle):
    min_dist = float('inf')
    for i in range(start_angle, end_angle + 1):
        if i < len(donnees_lidar) and donnees_lidar[i] > 0: min_dist = min(min_dist, donnees_lidar[i])
    return min_dist if min_dist != float('inf') else 0

def detect_sharp_turn(donnees_lidar):
    left_distance = get_min_distance(donnees_lidar, 45, 135)
    right_distance = get_min_distance(donnees_lidar, 225, 315)
    front_distance = get_min_distance(donnees_lidar, 350, 10)
    return abs(left_distance - right_distance) > 1.5 and 1.0 < front_distance < 4.0, left_distance, right_distance

def normalize_angle(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

# Position controller
def update_pose(yaw, speed_kmh, dt):
    global robot_pose, last_yaw, trajectory_history
    if last_yaw is None: last_yaw = yaw; return
    delta_yaw = normalize_angle(yaw - last_yaw)
    robot_pose['theta'] += delta_yaw
    dist = (speed_kmh / 3.6) * (dt / 1000.0)
    robot_pose['x'] += dist * math.cos(robot_pose['theta'])
    robot_pose['y'] += dist * math.sin(robot_pose['theta'])
    trajectory_history.append((robot_pose['x'], robot_pose['y']))
    last_yaw = yaw

def world_to_grid(x, y): return int(x/MAP_RESOLUTION)+MAP_ORIGIN, int(y/MAP_RESOLUTION)+MAP_ORIGIN

def update_occupancy_grid(donnees_lidar):
    global occupancy_grid
    rx, ry, rt = robot_pose['x'], robot_pose['y'], robot_pose['theta']
    for ai in range(0, 360, 2):
        if ai >= len(donnees_lidar): continue
        d = donnees_lidar[ai]
        if d <= 0.1 or d > 10.0: continue
        la = math.radians(ai)
        ox, oy = rx + d*math.cos(rt+la), ry + d*math.sin(rt+la)
        gx, gy = world_to_grid(ox, oy)
        if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE: occupancy_grid[gy,gx] = max(0.0, occupancy_grid[gy,gx]-0.15)
        rgx, rgy = world_to_grid(rx, ry)
        for s in range(1, min(int(d/MAP_RESOLUTION), 50)):
            t = s / (d/MAP_RESOLUTION)
            fx, fy = int(rgx+t*(gx-rgx)), int(rgy+t*(gy-rgy))
            if 0 <= fx < MAP_SIZE and 0 <= fy < MAP_SIZE: occupancy_grid[fy,fx] = min(1.0, occupancy_grid[fy,fx]+0.03)

def predict_curve(donnees_lidar, current_speed, current_angle):
    global curve_detected_ahead
    fl = get_min_distance(donnees_lidar, 330, 345)
    fc = get_min_distance(donnees_lidar, 345, 15)
    fr = get_min_distance(donnees_lidar, 15, 30)
    ls = get_min_distance(donnees_lidar, 60, 120)
    rs = get_min_distance(donnees_lidar, 240, 300)
    curve_detected_ahead = (0 < fc < 3.0 and abs(ls-rs) > 0.6) or abs(fl-fr) > 0.8 or abs(current_angle) > 0.28
    return curve_detected_ahead

def detect_loop():
    global loop_closure_detected
    if len(trajectory_history) < 20: return False
    cx, cy = robot_pose['x'], robot_pose['y']
    for i in range(len(trajectory_history)-10):
        hx, hy = trajectory_history[i]
        if math.sqrt((cx-hx)**2+(cy-hy)**2) < 3.0: loop_closure_detected = True; return True
    loop_closure_detected = False
    return False

def detect_contresens(gyro_data, current_speed):
    global contresens_counter
    if len(trajectory_history) < CONTRESENS_HISTORY_MIN or current_speed < 1.5: contresens_counter = 0; return False
    if abs(gyro_data[2]) > SHARP_TURN_YAW_RATE: contresens_counter = max(0, contresens_counter-2); return False
    cx, cy, ct = robot_pose['x'], robot_pose['y'], robot_pose['theta']
    hx, hy = math.cos(ct), math.sin(ct)
    fx, fy = cx + hx*1.5, cy + hy*1.5
    cr = len(trajectory_history) - 20
    for i in range(max(0, cr-100), cr):
        px, py = trajectory_history[i]
        if math.sqrt((fx-px)**2+(fy-py)**2) < 1.0 and i > 0:
            ppx, ppy = trajectory_history[i-1]
            dx, dy = px-ppx, py-ppy
            n = math.sqrt(dx*dx+dy*dy)
            if n > 0.01 and (hx*dx/n + hy*dy/n) < -0.5:
                contresens_counter += 1
                return contresens_counter >= CONTRESENS_CONFIRM_FRAMES
    contresens_counter = max(0, contresens_counter-1)
    return False

# Coeur: controleur de vitesse
def speed_control(donnees_lidar, roll, pitch, gyro_data, current_angle):
    global emergency_mode, curve_detected_ahead, loop_closure_detected
    fm = get_min_distance(donnees_lidar, 330, 30)
    cf = get_min_distance(donnees_lidar, 345, 15)
    vf = get_min_distance(donnees_lidar, 350, 10)
    yr = abs(gyro_data[2])
    tsf = max(0.5, 1.0 - yr*0.8)
    cp = 0.95 if curve_detected_ahead else 1.0
    lp = 0.85 if loop_closure_detected else 1.0
    if emergency_mode: bs = 3.5
    elif 0 < vf < 0.8: bs = 0.8
    elif 0 < cf < collision_threshold: bs = 2.0*tsf
    elif 0 < fm < obstacle_threshold: bs = (4.0 + fm/obstacle_threshold*4.0)*tsf
    elif fm > safe_distance: bs = min(18.0, 9.0 + (fm-safe_distance)*2.5)*tsf
    else: bs = 7.5*tsf
    bs *= cp*lp
    if abs(current_angle) > 0.28: bs = min(bs, 3.5)
    return bs

def steering_control(donnees_lidar, base_angle, roll, gyro_data):
    global last_angle, angle_history, emergency_mode, emergency_counter, turn_detected, loop_closure_detected
    ra = base_angle
    ist, ld, rd = detect_sharp_turn(donnees_lidar)
    fd = get_min_distance(donnees_lidar, 350, 10)
    lf = get_min_distance(donnees_lidar, 30, 60)
    rf = get_min_distance(donnees_lidar, 300, 330)
    ls = get_min_distance(donnees_lidar, 60, 120)
    rs = get_min_distance(donnees_lidar, 240, 300)
    if ist:
        turn_detected = True
        if ld > rd: ra = max(ra, -0.3); ra = -0.23 if abs(ra) < 0.15 else ra
        else: ra = min(ra, 0.3); ra = 0.23 if abs(ra) < 0.15 else ra
    else: turn_detected = False
    if 0 < fd < 1.2: emergency_mode, emergency_counter = True, 15; ra = -0.34 if lf > rf else 0.34
    if not ist and not emergency_mode:
        if abs(ra) > 0.1: ra *= 2.2 if (ra > 0 and rs > ls+0.3) or (ra < 0 and ls > rs+0.3) else 1.6
        elif abs(ra) > 0.05: ra *= 1.4
    if 0 < fd < 2.0:
        if ls > 2.5 and ls > rs+0.5: ra = min(ra-0.12, ra*1.5) if ra < 0 else -0.18
        elif rs > 2.5 and rs > ls+0.5: ra = max(ra+0.12, ra*1.5) if ra > 0 else 0.18
    if abs(gyro_data[2]) > 1.5: ra *= 0.6
    if emergency_mode: emergency_counter -= 1; emergency_mode = emergency_counter > 0
    ra = max(-0.34, min(0.34, ra))
    angle_history.append(ra)
    if ist or emergency_mode:
        if len(angle_history) > 1: angle_history.pop(0)
        sa = ra
    else:
        if len(angle_history) > 2: angle_history.pop(0)
        if len(angle_history) == 1: sa = angle_history[0]
        elif len(angle_history) == 2: sa = angle_history[0]*0.35 + angle_history[-1]*0.65
        else: sa = angle_history[0]*0.2 + angle_history[1]*0.3 + angle_history[-1]*0.5
    mtc = 0.30 if loop_closure_detected else (0.25 if ist or emergency_mode else 0.12)
    if abs(sa - last_angle) > mtc: sa = last_angle + mtc if sa > last_angle else last_angle - mtc
    last_angle = sa
    return sa

#----Main----
while driver.step() != -1:
    speed = driver.getTargetCruisingSpeed()
    while True:
        donnees_lidar = lidar.getRangeImage()
        imu_data = imu.getRollPitchYaw()
        gyro_data = gyro.getValues()
        print_counter += 1
        if print_counter >= PRINT_INTERVAL:
            print(f"Yaw:{math.degrees(imu_data[2]):7.2f}° YawRate:{gyro_data[2]:5.2f} Speed:{speed:5.1f} Angle:{math.degrees(angle):6.2f}° Uturn:{uturn_mode}({uturn_state}) Pos:({robot_pose['x']:.1f},{robot_pose['y']:.1f})")
            print_counter = 0
        currentKey = keyboard.getKey()
        if currentKey == -1: break
        if currentKey in [ord('a'), ord('A')] and not modeAuto: modeAuto, modeManuel, uturn_mode = True, False, False
        elif currentKey in [ord('t'), ord('T')] and not uturn_mode and (modeAuto or modeManuel):
            uturn_mode, uturn_state = True, 0; print("--- Demi-tour manuel ---")
        if modeManuel and not uturn_mode:
            if currentKey == keyboard.UP: speed += 0.2
            elif currentKey == keyboard.DOWN: speed -= 0.2
            elif currentKey == keyboard.LEFT: angle -= 0.04
            elif currentKey == keyboard.RIGHT: angle += 0.04
    if not modeManuel and not modeAuto: speed, angle = 0, 0

    # Correction contresens
    if uturn_mode:
        yaw = imu_data[2]
        front = get_min_distance(donnees_lidar, 350, 10)
        rear = get_min_distance(donnees_lidar, 170, 190)
        left = get_min_distance(donnees_lidar, 60, 120)
        right = get_min_distance(donnees_lidar, 240, 300)
        
        # Init - choisir direction et calculer cible
        if uturn_state == 0:
            speed = 0
            uturn_initial_yaw = yaw
            uturn_direction = 1 if left > right else -1  # 1=gauche, -1=droite
            uturn_target_yaw = normalize_angle(yaw + math.pi)  # 180° opposé
            uturn_timer = 0
            uturn_state = 1
            print(f"Uturn: dir={'gauche' if uturn_direction==1 else 'droite'} target={math.degrees(uturn_target_yaw):.1f}°")
        
        # State 1 inverse de la direction finale
        elif uturn_state == 1:
            angle = -uturn_direction * maxangle  # Braquer inverse
            speed = -3.0
            uturn_timer += 1
            yaw_diff = abs(normalize_angle(yaw - uturn_initial_yaw))
 
            if yaw_diff > math.radians(60) or (0 < rear < 0.4) or uturn_timer > 80:
                uturn_state = 2
                uturn_timer = 0
        
        # State 2 Avancer en braquant vers direction cible
        elif uturn_state == 2:
            angle = uturn_direction * maxangle
            speed = 4.0
            uturn_timer += 1
            yaw_to_target = abs(normalize_angle(yaw - uturn_target_yaw))

            if yaw_to_target < math.radians(30) or (0 < front < 0.5) or uturn_timer > 80:
                if yaw_to_target < math.radians(30):
                    uturn_state = 4  
                else:
                    uturn_state = 3  
                    uturn_timer = 0
        
        # State 3 Reculer encore si nécessaire 
        elif uturn_state == 3:
            angle = -uturn_direction * maxangle
            speed = -3.0
            uturn_timer += 1
            yaw_to_target = abs(normalize_angle(yaw - uturn_target_yaw))
            if yaw_to_target < math.radians(50) or (0 < rear < 0.4) or uturn_timer > 60:
                uturn_state = 4
                uturn_timer = 0
        
        # State 4 Finalisation - aligner et reprendre
        elif uturn_state == 4:
            yaw_to_target = normalize_angle(uturn_target_yaw - yaw)
            if abs(yaw_to_target) > math.radians(10):
                angle = maxangle if yaw_to_target > 0 else -maxangle
                speed = 2.0
                uturn_timer += 1
                if uturn_timer > 50: uturn_state = 5 
            else:
                uturn_state = 5
        
        # State 5: Terminé
        elif uturn_state == 5:
            uturn_mode = False
            last_yaw = yaw
            trajectory_history.clear()
            contresens_counter = 0
            angle = 0
            print("--- Demi-tour terminé ---")

    elif modeAuto:
        roll, pitch, yaw = imu_data[0], imu_data[1], imu_data[2]
        update_pose(yaw, speed, sensorTimeStep)
        slam_update_counter += 1
        if slam_update_counter >= SLAM_UPDATE_INTERVAL:
            update_occupancy_grid(donnees_lidar)
            predict_curve(donnees_lidar, speed, angle)
            detect_loop()
            slam_update_counter = 0
        if detect_contresens(gyro_data, speed):
            print("Contresens! Demi-tour auto ON")
            uturn_mode, uturn_state = True, 0
        else:
            base_angle = donnees_lidar[240] - donnees_lidar[120]
            speed = speed_control(donnees_lidar, roll, pitch, gyro_data, angle)
            angle = steering_control(donnees_lidar, base_angle, roll, gyro_data)
            if speed < 2.0 and abs(angle) < 0.08:
                ls = get_min_distance(donnees_lidar, 60, 120)
                rs = get_min_distance(donnees_lidar, 240, 300)
                if ls > rs and ls > 1.5: angle = -0.3
                elif rs > ls and rs > 1.5: angle = 0.3

    speed = max(-maxSpeed, min(maxSpeed, speed))
    angle = max(-maxangle, min(maxangle, angle))
    driver.setCruisingSpeed(speed)
    driver.setSteeringAngle(angle)