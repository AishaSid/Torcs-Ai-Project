import gymnasium as gym
from gymnasium import spaces
import numpy as np
import msgParser
import carState
import carControl

class TORCSEnv(gym.Env):
    def __init__(self):
        super(TORCSEnv, self).__init__()
        
        # Initialize TORCS components
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Define action space (continuous)
        # [steer, accel, brake] where:
        # steer: [-1, 1]
        # accel: [0, 1]
        # brake: [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space
        # 32 features from preprocessing:
        # - 19 track sensors
        # - TrackPos, Angle, SpeedX, SpeedY, SpeedZ
        # - Gear, RPM, MinOpponent
        # - speed_magnitude, acceleration, turn_sharpness
        # - track_pos_stability, speed_stability
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.max_steps = 1000
        self.last_reward = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.last_reward = 0
        
        # Initialize car state
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        # Return initial observation
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment."""
        self.current_step += 1
        
        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply actions
        steer, accel, brake = action
        self.control.setSteer(float(steer))
        self.control.setAccel(float(accel))
        self.control.setBrake(float(brake))
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = False  # We don't use truncation in this environment
        
        # Additional info
        info = {
            'speed': float(self.state.getSpeedX() or 0.0),
            'track_pos': float(self.state.getTrackPos() or 0.0),
            'damage': float(self.state.getDamage() or 0.0),
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Convert current state to observation array."""
        # Get track sensors
        track = self.state.getTrack()
        if track is None:
            track = [0] * 19
            
        # Get other state variables with default values if None
        track_pos = self.state.getTrackPos() or 0.0
        angle = self.state.getAngle() or 0.0
        speed_x = self.state.getSpeedX() or 0.0
        speed_y = self.state.getSpeedY() or 0.0
        speed_z = self.state.getSpeedZ() or 0.0
        gear = self.state.getGear() or 0
        rpm = self.state.getRpm() or 0
        
        # Calculate additional features
        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2 + speed_z**2)
        
        # Get opponents
        opponents = self.state.getOpponents()
        min_opponent = min(opponents) if opponents else 200
        
        # Combine all features
        observation = np.array([
            *track,  # 19 track sensors
            track_pos,
            angle,
            speed_x,
            speed_y,
            speed_z,
            gear,
            rpm,
            min_opponent,
            speed_magnitude,
            0.0,  # acceleration (would need previous state)
            0.0,  # turn_sharpness (would need previous state)
            0.0,  # track_pos_stability (would need previous state)
            0.0   # speed_stability (would need previous state)
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self):
        """Calculate reward based on current state."""
        # Get current state values with defaults
        speed = self.state.getSpeedX() or 0.0
        track_pos = self.state.getTrackPos() or 0.0
        damage = self.state.getDamage() or 0.0
        angle = self.state.getAngle() or 0.0
        
        # Speed reward (encourage high speed)
        speed_reward = speed / 100.0
        
        # Track position penalty (penalize being far from center)
        track_pos_penalty = -abs(track_pos)
        
        # Damage penalty
        damage_penalty = -damage / 100.0
        
        # Angle penalty (penalize large angles)
        angle_penalty = -abs(angle) / np.pi
        
        # Combine rewards
        reward = (
            speed_reward +
            track_pos_penalty +
            damage_penalty +
            angle_penalty
        )
        
        # Add reward for staying alive
        reward += 0.1
        
        # Ensure reward is a float
        return float(reward)
    
    def _is_done(self):
        """Check if episode should end."""
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Check if car is damaged
        damage = self.state.getDamage() or 0.0
        if damage > 1000:
            return True
        
        # Check if car is stuck (very low speed for too long)
        speed = self.state.getSpeedX() or 0.0
        if speed < 1.0 and self.current_step > 100:
            return True
        
        return False
    
    def render(self):
        """Render the environment."""
        pass  # TORCS handles rendering
    
    def close(self):
        """Clean up environment resources."""
        pass 