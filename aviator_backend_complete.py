# app.py - Complete Aviator Predictor Backend
import asyncio
import json
import logging
import os
import sqlite3
import websockets
import uvicorn
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
from contextlib import asynccontextmanager
from enum import Enum

# Configure logging
def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration"""
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

# Setup logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', './logs/aviator.log')
setup_logging(LOG_LEVEL, LOG_FILE)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Database settings
    DB_PATH: str = os.getenv('DB_PATH', './data/aviator.db')
    
    # Server settings
    WS_HOST: str = os.getenv('WS_HOST', '0.0.0.0')
    WS_PORT: int = int(os.getenv('WS_PORT', '8765'))
    WEB_HOST: str = os.getenv('WEB_HOST', '0.0.0.0')
    WEB_PORT: int = int(os.getenv('WEB_PORT', '8080'))
    
    # Prediction model settings
    PREDICTION_WINDOW_SIZE: int = int(os.getenv('PREDICTION_WINDOW_SIZE', '20'))
    PREDICTION_INTERVAL: int = int(os.getenv('PREDICTION_INTERVAL', '30'))
    
    # Performance settings
    MAX_CONNECTIONS: int = int(os.getenv('MAX_CONNECTIONS', '100'))
    CLEANUP_INTERVAL: int = int(os.getenv('CLEANUP_INTERVAL', '3600'))
    
    # Model parameters
    MODEL_CONFIG: Dict[str, Any] = {
        'window_size': PREDICTION_WINDOW_SIZE,
        'volatility_threshold': 0.3,
        'trend_weight': 0.25,
        'volatility_weight': 0.20,
        'momentum_weight': 0.20,
        'mean_reversion_weight': 0.15,
        'cycle_weight': 0.10,
        'variance_weight': 0.10,
        'confidence_threshold': 0.7,
        'prediction_bounds': (1.0, 50.0),
        'min_data_points': 5
    }

# Enums
class GameStatus(Enum):
    WAITING = "waiting"
    FLYING = "flying"
    CRASHED = "crashed"

class PredictionStatus(Enum):
    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"

# Data models
@dataclass
class GameRound:
    id: int
    multiplier: float
    timestamp: datetime
    duration: float
    status: GameStatus = GameStatus.CRASHED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'multiplier': self.multiplier,
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration,
            'status': self.status.value
        }

@dataclass
class PredictionFeatures:
    """Features used for prediction"""
    trend: float = 0.0
    volatility: float = 0.0
    momentum: float = 0.0
    mean_reversion: float = 0.0
    cycle_position: float = 0.0
    recent_variance: float = 0.0
    streak_count: int = 0
    time_factor: float = 0.0

@dataclass
class Prediction:
    round_id: int
    predicted_multiplier: float
    confidence: float
    timestamp: datetime
    actual_multiplier: Optional[float] = None
    accuracy: Optional[float] = None
    status: PredictionStatus = PredictionStatus.PENDING
    model_version: str = "2.0"
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'round_id': self.round_id,
            'predicted_multiplier': self.predicted_multiplier,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'actual_multiplier': self.actual_multiplier,
            'accuracy': self.accuracy,
            'status': self.status.value,
            'model_version': self.model_version,
            'features': self.features
        }

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    predictions_per_minute: int
    database_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'active_connections': self.active_connections,
            'predictions_per_minute': self.predictions_per_minute,
            'database_size': self.database_size
        }

class PredictionRequest(BaseModel):
    game_data: List[Dict]

# Custom Exceptions
class AviatorPredictorException(Exception):
    """Base exception for Aviator Predictor"""
    pass

class DatabaseException(AviatorPredictorException):
    """Database related exceptions"""
    pass

class PredictionException(AviatorPredictorException):
    """Prediction related exceptions"""
    pass

# Utility functions
def calculate_accuracy(predicted: float, actual: float) -> float:
    """Calculate prediction accuracy"""
    if predicted == 0:
        return 0.0
    return 1.0 - abs(predicted - actual) / max(predicted, actual)

def validate_multiplier(multiplier: float) -> bool:
    """Validate if multiplier is within acceptable range"""
    return 1.0 <= multiplier <= 100.0

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.prediction_count = 0
        self.last_reset = time.time()
    
    def record_request(self):
        """Record a new request"""
        self.request_count += 1
    
    def record_prediction(self):
        """Record a new prediction"""
        self.prediction_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate rates (per minute)
        time_since_reset = current_time - self.last_reset
        requests_per_minute = (self.request_count / time_since_reset) * 60 if time_since_reset > 0 else 0
        predictions_per_minute = (self.prediction_count / time_since_reset) * 60 if time_since_reset > 0 else 0
        
        return {
            'uptime_seconds': round(uptime, 2),
            'requests_per_minute': round(requests_per_minute, 2),
            'predictions_per_minute': round(predictions_per_minute, 2),
            'total_requests': self.request_count,
            'total_predictions': self.prediction_count
        }
    
    def reset_counters(self):
        """Reset rate counters"""
        self.request_count = 0
        self.prediction_count = 0
        self.last_reset = time.time()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
            
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.add(connection)
                
        # Remove disconnected connections
        for conn in disconnected:
            self.active_connections.discard(conn)

# Database Manager
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Game rounds table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS game_rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    multiplier REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    duration REAL NOT NULL,
                    status TEXT DEFAULT 'crashed'
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_id INTEGER NOT NULL,
                    predicted_multiplier REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    actual_multiplier REAL,
                    accuracy REAL,
                    status TEXT DEFAULT 'pending',
                    model_version TEXT DEFAULT '2.0',
                    features TEXT,
                    FOREIGN KEY (round_id) REFERENCES game_rounds (id)
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    correct_predictions INTEGER NOT NULL,
                    accuracy_rate REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    system_cpu REAL NOT NULL,
                    system_memory REAL NOT NULL
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rounds_timestamp ON game_rounds(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_round_id ON predictions(round_id)')
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    def add_game_round(self, multiplier: float, duration: float, status: str = 'crashed') -> int:
        """Add a new game round"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO game_rounds (multiplier, timestamp, duration, status)
                VALUES (?, ?, ?, ?)
            ''', (multiplier, datetime.now(), duration, status))
            conn.commit()
            return cursor.lastrowid
            
    def add_prediction(self, round_id: int, predicted_multiplier: float, confidence: float, 
                      features: Dict[str, Any] = None) -> int:
        """Add a new prediction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            features_json = json.dumps(features) if features else None
            cursor.execute('''
                INSERT INTO predictions (round_id, predicted_multiplier, confidence, timestamp, features)
                VALUES (?, ?, ?, ?, ?)
            ''', (round_id, predicted_multiplier, confidence, datetime.now(), features_json))
            conn.commit()
            return cursor.lastrowid
            
    def update_prediction_result(self, prediction_id: int, actual_multiplier: float):
        """Update prediction with actual result"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            accuracy = calculate_accuracy(predicted_multiplier, actual_multiplier)
            status = 'correct' if accuracy > 0.8 else 'incorrect'
            
            cursor.execute('''
                UPDATE predictions 
                SET actual_multiplier = ?, accuracy = ?, status = ?
                WHERE id = ?
            ''', (actual_multiplier, accuracy, status, prediction_id))
            conn.commit()
            
    def get_recent_rounds(self, limit: int = 100) -> List[GameRound]:
        """Get recent game rounds"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, multiplier, timestamp, duration, status
                FROM game_rounds
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rounds = []
            for row in cursor.fetchall():
                rounds.append(GameRound(
                    id=row[0],
                    multiplier=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    duration=row[3],
                    status=GameStatus(row[4]) if row[4] else GameStatus.CRASHED
                ))
            return rounds
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive prediction statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total_predictions = cursor.fetchone()[0]
            
            # Accuracy statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_with_results,
                    AVG(accuracy) as avg_accuracy,
                    AVG(confidence) as avg_confidence,
                    COUNT(CASE WHEN status = 'correct' THEN 1 END) as correct_count
                FROM predictions 
                WHERE actual_multiplier IS NOT NULL
            ''')
            stats = cursor.fetchone()
            
            # Recent performance (last 24 hours)
            cursor.execute('''
                SELECT 
                    COUNT(*) as recent_predictions,
                    AVG(accuracy) as recent_accuracy,
                    COUNT(CASE WHEN status = 'correct' THEN 1 END) as recent_correct
                FROM predictions 
                WHERE actual_multiplier IS NOT NULL 
                AND timestamp > datetime('now', '-1 day')
            ''')
            recent_stats = cursor.fetchone()
            
            # Best and current streak
            cursor.execute('''
                SELECT status FROM predictions 
                WHERE actual_multiplier IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''')
            statuses = [row[0] for row in cursor.fetchall()]
            
            current_streak = 0
            best_streak = 0
            temp_streak = 0
            
            for status in statuses:
                if status == 'correct':
                    temp_streak += 1
                    if temp_streak == len(statuses) - statuses.index(status):
                        current_streak = temp_streak
                    best_streak = max(best_streak, temp_streak)
                else:
                    temp_streak = 0
            
            return {
                'total_predictions': total_predictions,
                'predictions_with_results': stats[0] if stats[0] else 0,
                'correct_predictions': stats[3] if stats[3] else 0,
                'accuracy_rate': round(stats[1], 4) if stats[1] else 0,
                'average_confidence': round(stats[2], 4) if stats[2] else 0,
                'predictions_24h': recent_stats[0] if recent_stats[0] else 0,
                'accuracy_24h': round(recent_stats[1], 4) if recent_stats[1] else 0,
                'correct_24h': recent_stats[2] if recent_stats[2] else 0,
                'current_streak': current_streak,
                'best_streak': best_streak
            }
            
    def get_recent_predictions(self, limit: int = 50) -> List[Prediction]:
        """Get recent predictions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT round_id, predicted_multiplier, confidence, timestamp, 
                       actual_multiplier, accuracy, status, model_version, features
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            predictions = []
            for row in cursor.fetchall():
                features = json.loads(row[8]) if row[8] else {}
                predictions.append(Prediction(
                    round_id=row[0],
                    predicted_multiplier=row[1],
                    confidence=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    actual_multiplier=row[4],
                    accuracy=row[5],
                    status=PredictionStatus(row[6]) if row[6] else PredictionStatus.PENDING,
                    model_version=row[7] or "2.0",
                    features=features
                ))
            return predictions

# Advanced Prediction Engine
class AdvancedPredictionEngine:
    """Advanced prediction engine with multiple algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_weights = {
            'trend': config.get('trend_weight', 0.25),
            'volatility': config.get('volatility_weight', 0.20),
            'momentum': config.get('momentum_weight', 0.20),
            'mean_reversion': config.get('mean_reversion_weight', 0.15),
            'cycle': config.get('cycle_weight', 0.10),
            'variance': config.get('variance_weight', 0.10)
        }
        
        self.window_size = config.get('window_size', 20)
        self.min_data_points = config.get('min_data_points', 5)
        self.prediction_bounds = config.get('prediction_bounds', (1.0, 50.0))
        
    def extract_features(self, rounds: List[GameRound]) -> PredictionFeatures:
        """Extract features from recent rounds"""
        if len(rounds) < self.min_data_points:
            return PredictionFeatures()
            
        multipliers = [round.multiplier for round in rounds[:self.window_size]]
        
        features = PredictionFeatures()
        
        # Calculate trend
        features.trend = self._calculate_trend(multipliers)
        
        # Calculate volatility
        features.volatility = self._calculate_volatility(multipliers)
        
        # Calculate momentum
        features.momentum = self._calculate_momentum(multipliers)
        
        # Calculate mean reversion tendency
        features.mean_reversion = self._calculate_mean_reversion(multipliers)
        
        # Calculate cycle position
        features.cycle_position = self._calculate_cycle_position(multipliers)
        
        # Calculate recent variance
        features.recent_variance = self._calculate_recent_variance(multipliers)
        
        # Calculate streak
        features.streak_count = self._calculate_streak(multipliers)
        
        # Time-based factor
        features.time_factor = self._calculate_time_factor(rounds)
        
        return features
    
    def predict_next_multiplier(self, recent_rounds: List[GameRound]) -> Prediction:
        """Predict the next multiplier using advanced algorithms"""
        if len(recent_rounds) < self.min_data_points:
            return Prediction(
                round_id=recent_rounds[0].id + 1 if recent_rounds else 1,
                predicted_multiplier=2.0,
                confidence=0.1,
                timestamp=datetime.now(),
                features={'insufficient_data': True}
            )
        
        # Extract features
        features = self.extract_features(recent_rounds)
        
        # Generate predictions using different algorithms
        predictions = []
        
        # Trend-based prediction
        trend_pred = self._trend_prediction(recent_rounds, features)
        predictions.append(('trend', trend_pred, self.model_weights['trend']))
        
        # Volatility-based prediction
        volatility_pred = self._volatility_prediction(recent_rounds, features)
        predictions.append(('volatility', volatility_pred, self.model_weights['volatility']))
        
        # Momentum-based prediction
        momentum_pred = self._momentum_prediction(recent_rounds, features)
        predictions.append(('momentum', momentum_pred, self.model_weights['momentum']))
        
        # Mean reversion prediction
        mean_reversion_pred = self._mean_reversion_prediction(recent_rounds, features)
        predictions.append(('mean_reversion', mean_reversion_pred, self.model_weights['mean_reversion']))
        
        # Cycle-based prediction
        cycle_pred = self._cycle_prediction(recent_rounds, features)
        predictions.append(('cycle', cycle_pred, self.model_weights['cycle']))
        
        # Variance-based prediction
        variance_pred = self._variance_prediction(recent_rounds, features)
        predictions.append(('variance', variance_pred, self.model_weights['variance']))
        
        # Ensemble prediction
        final_prediction = self._ensemble_prediction(predictions)
        
        # Calculate confidence
        confidence = self._calculate_confidence(features, predictions)
        
        # Apply bounds
        final_prediction = max(self.prediction_bounds[0], 
                             min(final_prediction, self.prediction_bounds[1]))
        
        return Prediction(
            round_id=recent_rounds[0].id + 1 if recent_rounds else 1,
            predicted_multiplier=round(final_prediction, 2),
            confidence=round(confidence, 3),
            timestamp=datetime.now(),
            features=asdict(features)
        )
    
    def _calculate_trend(self, multipliers: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(multipliers) < 3:
            return 0.0
        
        x = np.arange(len(multipliers))
        y = np.array(multipliers)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_volatility(self, multipliers: List[float]) -> float:
        """Calculate volatility (standard deviation)"""
        if len(multipliers) < 2:
            return 0.0
        return np.std(multipliers)
    
    def _calculate_momentum(self, multipliers: List[float]) -> float:
        """Calculate momentum based on recent changes"""
        if len(multipliers) < 3:
            return 0.0
        
        changes = np.diff(multipliers)
        momentum = np.mean(changes[-3:])
        return momentum
    
    def _calculate_mean_reversion(self, multipliers: List[float]) -> float:
        """Calculate mean reversion tendency"""
        if len(multipliers) < 5:
            return 0.0
        
        mean_val = np.mean(multipliers)
        current_val = multipliers[0]  # Most recent
        
        # Calculate deviation from mean
        deviation = current_val - mean_val
        
        # Mean reversion factor (negative means tendency to revert)
        return -deviation / mean_val
    
    def _calculate_cycle_position(self, multipliers: List[float]) -> float:
        """Calculate position in cycle (0-1)"""
        if len(multipliers) < 8:
            return 0.5
        
        # Simple cycle detection using autocorrelation
        try:
            autocorr = np.correlate(multipliers, multipliers, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peak (excluding lag 0)
            peaks = np.where(autocorr[1:] > np.mean(autocorr[1:]) + np.std(autocorr[1:]))[0]
            
            if len(peaks) > 0:
                cycle_length = peaks[0] + 1
                position = len(multipliers) % cycle_length
                return position / cycle_length
        except:
            pass
        
        return 0.5
    
    def _calculate_recent_variance(self, multipliers: List[float]) -> float:
        """Calculate recent variance"""
        if len(multipliers) < 5:
            return 0.0
        
        recent = multipliers[:5]
        return np.var(recent)
    
    def _calculate_streak(self, multipliers: List[float]) -> int:
        """Calculate current streak of high/low multipliers"""
        if len(multipliers) < 2:
            return 0
        
        mean_val = np.mean(multipliers)
        current_val = multipliers[0]
        
        streak = 0
        is_high = current_val > mean_val
        
        for mult in multipliers:
            if (mult > mean_val) == is_high:
                streak += 1
            else:
                break
        
        return streak if is_high else -streak
    
    def _calculate_time_factor(self, rounds: List[GameRound]) -> float:
        """Calculate time-based factor"""
        if len(rounds) < 2:
            return 0.0
        
        # Calculate average time between rounds
        time_diffs = []
        for i in range(len(rounds) - 1):
            diff = (rounds[i].timestamp - rounds[i+1].timestamp).total_seconds()
            time_diffs.append(diff)
        
        if time_diffs:
            avg_time = np.mean(time_diffs)
            return min(avg_time / 60.0, 1.0)  # Normalize to 1 minute
        
        return 0.0
    
    def _trend_prediction(self, rounds: List[GameRound], features: PredictionFeatures) -> float:
        """Trend-based prediction"""
        multipliers = [round.multiplier for round in rounds[:10]]
        base = np.mean(multipliers)
        
        # Apply trend
        trend_adjustment = features.trend * 2.0
        return base + trend_adjustment
    
    def _volatility_prediction(self, rounds: List[GameRound], features: PredictionFeatures) -> float:
        """Volatility-based prediction"""
        multipliers = [round.multiplier for round in rounds[:10]]
        base = np.mean(multipliers)
        
        # High volatility suggests more extreme values
        volatility_adjustment = features.volatility * 0.5
        return base + volatility_adjustment
    
    def _momentum_prediction(self, rounds: List[GameRound], features: PredictionFeatures) -> float:
        """Momentum-based prediction"""
        multipliers = [round.multiplier for round in rounds[:5]]
        base = multipliers[0]  # Most recent
        
        # Apply momentum
        momentum_adjustment = features.momentum * 1.5
        return base + momentum_adjustment
    
    def _mean_reversion_prediction(self, rounds: List[GameRound], features: PredictionFeatures) -> float:
        """Mean reversion prediction"""
        multipliers = [round.multiplier for round in rounds[:15]]
        mean_val = np.mean(multipliers)
        
        # Apply mean reversion
        reversion_adjustment = features.mean_reversion * mean_val
        return mean_val + reversion_adjustment
    
    def _cycle_prediction(self, rounds: List[GameRound], features: PredictionFeatures) -> float:
        """Cycle-based prediction"""
        multipliers = [round.multiplier for round in rounds[:10]]
        base = np.mean(multipliers)
        
        # Apply cycle position
        cycle_adjustment = np.sin(features.cycle_position * 2 * np.pi) * 0.5
        return base + cycle_adjustment
    
    def _variance_prediction(self, rounds: List[GameRound], features: PredictionFeatures) -> float:
        """Variance-based prediction"""
        multipliers = [round.multiplier for round in rounds[:10]]
        base = np.mean(multipliers)
        
        # Apply variance adjustment
        variance_adjustment = features.recent_variance * 0.3
        return base + variance_adjustment
    
    def _ensemble_prediction(self, predictions: List[Tuple[str, float, float]]) -> float:
        """Combine predictions using weighted ensemble"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, pred, weight in predictions:
            weighted_sum += pred * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        
        return 2.0  # Default
    
    def _calculate_confidence(self, features: PredictionFeatures, predictions: List[Tuple[str, float, float]]) -> float:
        """Calculate prediction confidence"""
        # Base confidence on data quality
        base_confidence = 0.6
        
        # Adjust for volatility (lower volatility = higher confidence)
        volatility_factor = max(0.1, 1.0 - features.volatility * 0.5)
        
        # Adjust for trend consistency
        trend_factor = max(0.3, 1.0 - abs(features.trend) * 0.2)
        
        # Adjust for prediction agreement
        pred_values = [pred[1] for pred in predictions]
        agreement_factor = max(0.2, 1.0 - np.std(pred_values) * 0.1)
        
        # Combine factors
        confidence = base_confidence * volatility_factor * trend_factor * agreement_factor
        
        return max(0.1, min(confidence, 0.95))

# Global instances
config = Config()
db_manager = DatabaseManager(config.DB_PATH)
predictor = AdvancedPredictionEngine(config.MODEL_CONFIG)
connection_manager = ConnectionManager()
performance_monitor = PerformanceMonitor()

# FastAPI app
app = FastAPI(title="Aviator Predictor API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket