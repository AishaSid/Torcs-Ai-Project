import msgParser
import carState
import carControl
import csv
import threading
from pynput import keyboard
# RL imports
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
from torcs_env import TORCSEnv

class Driver:
    '''
    A driver object for the SCRC
    '''
    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        
        self.manual_mode = False  # Default to RL mode
        self.manual_controls = {"steer": 0.0, "accel": 0.0, "gear": 1}  # Default manual controls
        
        # RL model integration
        self.rl_mode = True  # RL mode enabled by default
        self.rl_model = None
        self.rl_env = None
        self.rl_obs = None
        if self.rl_mode:
            def make_env():
                env = TORCSEnv()
                env = Monitor(env)
                return env
            self.rl_env = DummyVecEnv([make_env])
            self.rl_env = VecNormalize.load("logs/vec_normalize.pkl", self.rl_env)
            self.rl_env.training = False
            self.rl_env.norm_reward = False
            self.rl_model = SAC.load("logs/final_model.zip")
            self.rl_obs = self.rl_env.reset()
        
        # Start keyboard listener in a separate thread
        self.listener_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.listener_thread.start()
        
        # Open CSV file and write the header only once
        #with open("Road_CG1", "w", newline="") as file:
        #    writer = csv.writer(file)
        #    writer.writerow([
        #        "Angle", "SpeedX", "SpeedY", "SpeedZ",
        #        "Gear", "RPM", "Fuel", "TrackPos", "Distance Raced"    ])

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for _ in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        if self.manual_mode:
            self.control.setSteer(self.manual_controls["steer"])
            self.control.setAccel(self.manual_controls["accel"])
            self.control.setGear(self.manual_controls["gear"])
        elif self.rl_mode and self.rl_model is not None:
            try:
                obs = self._get_rl_observation()
                print("Observation shape:", obs.shape)
                print("Observation sample:", obs[0][:5])  # Print first 5 values
                
                action, _ = self.rl_model.predict(obs, deterministic=True)
                print("Raw action:", action)
                
                # Scale actions to appropriate ranges
                steer = np.clip(float(action[0][0]), -1.0, 1.0)  # Steer: [-1, 1]
                accel = np.clip(float(action[0][1]), 0.0, 1.0)   # Accel: [0, 1]
                brake = np.clip(float(action[0][2]), 0.0, 1.0)   # Brake: [0, 1]
                
                print(f"Raw Steer: {steer:.2f}, Raw Accel: {accel:.2f}, Raw Brake: {brake:.2f}")
                
                # Prioritize accel or brake, not both
                if accel >= brake:
                    # If acceleration is greater or equal, use only acceleration
                    brake = 0.0
                else:
                    # If braking is greater, use only braking
                    accel = 0.0
                
                print(f"Final Steer: {steer:.2f}, Final Accel: {accel:.2f}, Final Brake: {brake:.2f}")
                
                # Apply actions
                self.control.setSteer(steer)
                self.control.setAccel(accel)
                self.control.setBrake(brake)
                
                # Implement proper gear control
                rpm = self.state.getRpm()
                current_gear = self.state.getGear()
                
                # Gear shifting logic with more aggressive thresholds
                if rpm > 6000 and current_gear < 6:  # Shift up at high RPM
                    self.control.setGear(current_gear + 1)
                elif rpm < 2000 and current_gear > 1:  # Shift down at low RPM
                    self.control.setGear(current_gear - 1)
                elif current_gear == 0:  # If stuck in neutral, shift to first gear
                    self.control.setGear(1)
                
                # Print current state
                print(f"Speed: {self.state.getSpeedX():.2f}, TrackPos: {self.state.getTrackPos():.2f}")
                print(f"Gear: {self.state.getGear()}, RPM: {self.state.getRpm()}")
            except Exception as e:
                print(f"Error in RL control: {str(e)}")
                # Fallback to manual control if RL fails
                self.control.setSteer(0.0)
                self.control.setAccel(0.5)  # Set some default acceleration
                self.control.setBrake(0.0)
        else:
            self.steer()
            self.gear()
            self.speed()
        
        # Save car state data to CSV file
        self.save_data()  
        
        return self.control.toMsg()

    def _get_rl_observation(self):
        # Build the observation as expected by your RL model
        track = self.state.getTrack() or [0]*19
        track_pos = self.state.getTrackPos() or 0.0
        angle = self.state.getAngle() or 0.0
        speed_x = self.state.getSpeedX() or 0.0
        speed_y = self.state.getSpeedY() or 0.0
        speed_z = self.state.getSpeedZ() or 0.0
        gear = self.state.getGear() or 0
        rpm = self.state.getRpm() or 0
        opponents = self.state.getOpponents()
        min_opponent = min(opponents) if opponents else 200
        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2 + speed_z**2)
        
        # Print raw state values for debugging
        print("\nRaw State Values:")
        print(f"Track: {track[:5]}...")  # First 5 track values
        print(f"TrackPos: {track_pos}")
        print(f"Angle: {angle}")
        print(f"Speed: {speed_x}")
        print(f"Gear: {gear}")
        print(f"RPM: {rpm}")
        
        obs = np.array([
            *track, track_pos, angle, speed_x, speed_y, speed_z,
            gear, rpm, min_opponent, speed_magnitude,
            0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        return obs.reshape(1, -1)
    
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        self.control.setSteer((angle - dist * 0.5) / self.steer_lock)
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()

        if self.prev_rpm is None:
            up = True
        else:
            up = (self.prev_rpm - rpm) < 0

        if up and rpm > 7000:
            gear += 1

        if not up and rpm < 3000:
            gear -= 1

        self.control.setGear(gear)
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()

        if speed < self.max_speed:
            accel += 0.1
            accel = min(accel, 1.0)
        else:
            accel -= 0.1
            accel = max(accel, 0.0)

        self.control.setAccel(accel)
    
    def save_data(self):
        """Save all available car state data to a CSV file."""
        data = [
            # Positional and Orientation Data
            self.state.getAngle(),
            self.state.getTrackPos(),
            self.state.getDistFromStart(),
            self.state.getDistRaced(),
            self.state.getZ(),

            # Speed Components
            self.state.getSpeedX(),
            self.state.getSpeedY(),
            self.state.getSpeedZ(),

            # Vehicle Status
            self.state.getGear(),
            self.state.getRpm(),
            self.state.getFuel(),
            self.state.getDamage(),
            self.state.getRacePos(),
            self.state.getCurLapTime(),
            self.state.getLastLapTime(),

            # Additional Sensors
            self.state.getFocus(),
            self.state.getTrack(),
            self.state.getOpponents(),
            self.state.getWheelSpinVel(),

            # Car Control
            self.control.getSteer(),
            self.control.getAccel(),
            self.control.getBrake(),
            self.control.getClutch(),
            self.control.getFocus(),
            self.control.getGear(),
            self.control.getMeta()


        ]

        # Open the file in append mode
        with open("dirt-peugot.csv", "a", newline="") as file:
            writer = csv.writer(file)
            
            # If this is the first time writing, write the header
            if file.tell() == 0:
                header = [
                    "Angle", "TrackPos", "DistFromStart", "DistRaced", "Z",
                    "SpeedX", "SpeedY", "SpeedZ",
                    "Gear", "RPM", "Fuel", "Damage", "RacePos", 
                    "CurLapTime", "LastLapTime",
                    "Focus", "Track", "Opponents", "WheelSpinVel", 
                    "Steer", "Accel", "Brake", "Clutch", "ControlFocus",
                    "Gear", "Meta"
                ]
                writer.writerow(header)
            
            writer.writerow(data)
    def enable_manual_mode(self, enable):
        """Enable or disable manual mode."""
        self.manual_mode = enable
    
    def listen_keyboard(self):
        """Listen for keyboard inputs and update manual controls."""
        def on_press(key):
            try:
                if key.char == 'w':
                    self.manual_controls["accel"] = 1.0  # Accelerate
                elif key.char == 's':
                    self.manual_controls["accel"] = -1.0  # Brake
                elif key.char == 'a':
                    self.manual_controls["steer"] = 0.5  # Steer left
                elif key.char == 'd':
                    self.manual_controls["steer"] = -0.5  # Steer right
                elif key.char == 'm':
                    self.manual_controls["gear"] += 1  # Gear up
                elif key.char == 'n':
                    self.manual_controls["gear"] -= 1  # Gear down
                elif key.char == 'r':  # Toggle RL mode
                    self.manual_mode = not self.manual_mode
                    print(f"Switched to {'manual' if self.manual_mode else 'RL'} mode")
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char in ['w', 's']:
                    self.manual_controls["accel"] = 0.0  # Stop acceleration
                if key.char in ['a', 'd']:
                    self.manual_controls["steer"] = 0.0  # Stop steering
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
