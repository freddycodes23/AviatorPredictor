# Aviator Predictor - Full Stack Application

This project is a sophisticated, full-stack application designed for real-time prediction of outcomes in the "Aviator" game. It comprises a powerful **FastAPI backend** that handles data processing, prediction modeling, and real-time communication, along with a sleek and intuitive **HTML, CSS, and JavaScript frontend** for user interaction, data visualization, and control.

The system is engineered for performance, featuring an advanced prediction engine, asynchronous operations, and efficient database management to provide users with timely and accurate game insights.

-----

## Architecture Overview

The application is built on a client-server model:

  * **Backend (Python/FastAPI)**: Serves as the core engine. It connects to a database, runs prediction algorithms, manages WebSocket connections for real-time updates, and exposes a RESTful API for other interactions.
  * **Frontend (HTML/JS/CSS)**: Provides the user interface. It communicates with the backend via WebSockets to display live predictions, statistics, and logs. It allows users to interact with the system, adjust settings, and view historical data.

-----

## Backend Development

The backend is a robust API built with Python using the FastAPI framework. It is responsible for all the heavy lifting, including data storage, analysis, and prediction generation.

### Core Technologies

  * **FastAPI**: A modern, high-performance web framework for building APIs with Python.
  * **Uvicorn**: A lightning-fast ASGI server, used to run the FastAPI application.
  * **WebSockets**: For full-duplex, real-time communication between the backend and frontend.
  * **SQLite**: A self-contained, serverless database engine used for storing game rounds, predictions, and system metrics.
  * **NumPy**: For efficient numerical operations and calculations within the prediction engine.
  * **Pydantic**: For data validation and settings management.

### Backend Features

  * **Advanced Prediction Engine**: Utilizes a multi-faceted algorithm that considers trend, volatility, momentum, mean reversion, and cyclical patterns to predict game multipliers.
  * **Real-time WebSocket Communication**: Pushes live predictions, statistics, and system status updates to all connected frontend clients.
  * **Persistent Data Storage**: Uses SQLite to store historical game data and prediction results, enabling long-term analysis and model improvement.
  * **Comprehensive Statistics Tracking**: Calculates and provides detailed performance metrics, including prediction accuracy, confidence levels, and winning streaks.
  * **System Performance Monitoring**: Tracks and logs key system metrics like CPU/memory usage and prediction throughput.
  * **Asynchronous Operations**: Built entirely on Python's `asyncio` to handle a large number of concurrent connections and I/O operations efficiently.

### Backend Setup and Installation

To get the backend server running locally, follow these steps:

1.  **Prerequisites**:

      * **Python 3.8+** must be installed on your system.

2.  **Clone the Repository**:

    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

3.  **Set up a Virtual Environment**:
    It's highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies**:
    Create a `requirements.txt` file with the following content, which includes all necessary packages for the backend.

    ```
    fastapi
    uvicorn[standard]
    websockets
    numpy
    pydantic
    psutil
    aiofiles
    ```

    Then, install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Backend Server**:
    The backend code is in the `app.py` file. Use `uvicorn` to run the server. For development, the `--reload` flag is useful as it automatically restarts the server when code changes are detected.

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8765 --reload
    ```

      * `app:app`: Refers to the `app` instance of the `FastAPI` class inside the `app.py` file.
      * `--host 0.0.0.0`: Makes the server accessible on your local network.
      * `--port 8765`: Runs the server on port 8765, which the frontend expects.
      * `--reload`: Enables auto-reloading for development.

Once running, the backend will be live and listening for WebSocket connections and API requests on `http://localhost:8765`.

-----

## Frontend Development

The frontend is a single-page dashboard built with standard web technologies, ensuring broad compatibility and ease of use.

### Core Technologies

  * **HTML5**: Structures the dashboard's content and layout.
  * **CSS3**: Provides modern styling, gradients, shadows, and responsive design.
  * **JavaScript (ES6+)**: Powers all dynamic functionality, including WebSocket communication, DOM manipulation, event handling, and data visualization.

### Frontend Features

  * **Real-time Data Display**: Connects to the backend via WebSockets to show live predictions, confidence scores, and system status.
  * **Interactive Controls**: Allows users to request predictions, switch models, and adjust system settings.
  * **Manual Data Entry**: Provides an interface for manually inputting game data for analysis.
  * **Data Visualization**: Includes a chart to visualize the history of recent multipliers.
  * **System Log**: Displays real-time messages from the backend for monitoring and debugging.
  * **Data Management**: Users can export session data to a JSON file and reset the system state.

-----

## Full Stack Setup for Local Development

To run the complete application with both the frontend and backend working together, follow these steps:

1.  **Start the Backend Server**:
    Follow the "Backend Setup and Installation" steps above to get the `uvicorn` server running. Ensure it is active on `ws://localhost:8765`.

2.  **Run the Frontend Dashboard**:

      * Save the frontend code as an `index.html` file.
      * Open the `index.html` file in a modern web browser (like Chrome, Firefox, or Edge).

The dashboard will load and automatically attempt to connect to the running backend server. Once connected, the "Status" indicator will turn green, and you will start receiving real-time data. _ALGOUATION_
