# 🏠 House Price Prediction | Machine Learning Web Application

An interactive machine learning application that estimates residential property prices based on property characteristics, structural features, living area, property condition, and location-related factors.

This project demonstrates how machine learning can be applied to real estate valuation through a web-based interface. Users can enter property details, generate an estimated valuation, and explore additional property pricing indicators.

<p align="center">
  <a href="https://house-price-india-prediction-3.onrender.com/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20Application-brightgreen?style=for-the-badge&logo=render" alt="Live Demo">
  </a>
  <a href="https://github.com/Siddhesh030/House-Price-Prediction">
    <img src="https://img.shields.io/badge/GitHub-Source%20Code-black?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

---

## 🌐 Live Demo

**Try the application here:**

🔗 https://house-price-india-prediction-3.onrender.com/

The application provides an interactive interface for entering property information and obtaining a machine learning-based price estimate.

### What You Can Explore

- Enter residential property specifications.
- Specify the number of bedrooms and bathrooms.
- Provide living area and total lot area.
- Include basement and above-basement area measurements.
- Specify the property's construction year and renovation year.
- Select property condition and waterfront-view characteristics.
- Include design grade, view score, and nearby-school information.
- Generate an estimated property valuation.
- Review additional valuation indicators, including estimated price per square foot and monthly rent.

---

## 📌 Project Overview

House prices are influenced by multiple factors, including property size, structural characteristics, construction quality, age, renovations, and surrounding amenities.

Estimating property value manually can be difficult because these factors interact in different ways. This project addresses the problem using a supervised machine learning regression approach.

The application uses a trained **Linear Regression model** to estimate property prices from numerical and categorical property-related inputs.

The objective is to build an accessible valuation tool that demonstrates the practical integration of machine learning, Python, model persistence, and an interactive web interface.

### 🎯 Project Objectives

- Develop a machine learning model for residential property price estimation.
- Use property characteristics as input features for price prediction.
- Apply a trained regression model to generate numerical predictions.
- Build an interactive web interface for user input and prediction results.
- Present additional property valuation indicators.
- Deploy the application online so users can access it through a browser.
- Demonstrate an end-to-end machine learning deployment workflow.

---

## ✨ Key Features

### 1. 🏡 Interactive Property Valuation

Users can enter property specifications through an interactive interface and calculate an estimated property value.

### 2. 📐 Property Size Analysis

The application accepts several measurements that describe the size and layout of a property, including:

- Living area
- Total lot area
- Above-basement area
- Basement area
- Renovated lot area

These characteristics provide information about the property's usable space and physical dimensions.

### 3. 🛏️ Structural Characteristics

Users can specify important structural attributes, such as:

- Number of bedrooms
- Number of bathrooms
- Number of floors
- Property condition
- Design grade

These variables help represent differences in property layout, condition, and construction quality.

### 4. 🏗️ Construction and Renovation Information

The application accepts construction and renovation years.

This information can help the model distinguish between properties with different ages and renovation histories, depending on how these variables were represented during training.

### 5. 🌳 Location and Property Amenities

The application includes additional property-related inputs, such as:

- Waterfront-view status
- View score
- Number of nearby schools
- Distance from the airport

These attributes represent environmental, accessibility, and amenity-related differences between properties.

### 6. 💰 Estimated Property Valuation

After the user submits the property details, the application displays an estimated property value generated using the trained regression model.

The result is a model-based estimate rather than a guaranteed selling price.

### 7. 📊 Additional Valuation Indicators

The application interface also provides supplementary valuation information, including:

- Estimated price per square foot
- Estimated monthly rent
- Property quality tier
- Key feature weight breakdown

The interpretation of these indicators depends on the calculations and presentation logic implemented in the application.

### 8. 🚀 Live Web Deployment

The application is deployed on Render and can be accessed without requiring users to install Python or configure the project locally.

---

## 🧠 Machine Learning Approach

### Supervised Learning

This project uses supervised machine learning because the model learns relationships between property-related input features and known property prices from training data.

### Regression

House price prediction is a regression problem because the target is a continuous numerical value.

Unlike classification models, which assign observations to categories, regression models estimate numerical outcomes.

### Linear Regression

The application uses a trained Linear Regression model stored in the repository as `linear_model.pkl`.

Linear Regression estimates a numerical target by learning coefficients for the input variables.

A simplified representation of the model is:

\[
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n
\]

Where:

- \(\hat{y}\) represents the predicted property price.
- \(\beta_0\) represents the intercept.
- \(\beta_1, \beta_2, \ldots, \beta_n\) represent learned model coefficients.
- \(x_1, x_2, \ldots, x_n\) represent the property input features.

The trained model uses the feature representation expected by the application to produce a price estimate.

**Important:** The model's actual evaluation metrics, training-data characteristics, and preprocessing procedure should be documented using the original training workflow rather than assumed values.

---

## 🔄 Project Workflow

The application follows a machine learning prediction workflow.

1. **User Input:** The user enters property specifications through the web interface.
2. **Input Processing:** The application collects the submitted property characteristics.
3. **Feature Preparation:** The application prepares the input in the format required by the trained model.
4. **Model Loading:** The application uses the saved `linear_model.pkl` artifact.
5. **Price Prediction:** The trained Linear Regression model generates a numerical estimate.
6. **Result Presentation:** The interface displays the estimated valuation and additional pricing indicators.
7. **User Interpretation:** The user reviews the estimate as an initial reference for property valuation.

### Workflow Diagram

```text
              User
               |
               v
    Enter Property Specifications
               |
               v
       Input Processing
               |
               v
       Feature Preparation
               |
               v
     Trained Linear Regression
               |
               v
       Estimated Property Price
               |
               v
    Additional Valuation Indicators
               |
               v
        Results Interface
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Machine Learning | Property price estimation |
| Linear Regression | Predictive regression algorithm |
| Pickle-based model artifact | Storage of the trained model |
| HTML, CSS, and JavaScript, where implemented | Web interface and interactive elements |
| Render | Cloud deployment and application hosting |
| Git | Version control |
| GitHub | Source code management |

The repository contains the application code, saved model artifact, and dependency specification.

The exact interface libraries and Python packages should be confirmed against `app.py` and `requirements.txt` before adding further technologies to this list.

---

## 📂 Project Structure

The repository currently contains the following main files:

```text
House-Price-Prediction/
│
├── app.py
│   └── Main application and prediction workflow
│
├── linear_model.pkl
│   └── Saved Linear Regression model
│
├── requirements.txt
│   └── Python dependencies required by the project
│
└── README.md
    └── Project documentation
```

### File Descriptions

**`app.py`**

Contains the main application logic. It is responsible for implementing the web application and connecting the user-facing prediction workflow to the trained model.

**`linear_model.pkl`**

Stores the trained regression model so the application can load it for predictions without retraining it every time.

**`requirements.txt`**

Lists the Python packages required to set up and run the application.

**`README.md`**

Documents the project, its purpose, functionality, technology stack, and setup instructions.

---

## ⚙️ Installation and Setup

Follow these steps to run the project on your local machine.

### Prerequisites

Install the following:

- Python
- Git
- pip, the Python package installer

A compatible Python version should be selected based on the dependencies specified in `requirements.txt`.

### Step 1: Clone the Repository

Open a terminal or command prompt and execute:

```bash
git clone https://github.com/Siddhesh030/House-Price-Prediction.git
```

### Step 2: Navigate to the Project Directory

```bash
cd House-Price-Prediction
```

### Step 3: Create a Virtual Environment

Creating a virtual environment helps isolate project dependencies.

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python app.py
```

Use the local address printed in the terminal to open the application in your browser.

**Note:** If `app.py` uses a framework-specific launch command rather than starting a web server directly, use the corresponding command defined by the project. For example, if the application is implemented with Streamlit, the command would be:

```bash
streamlit run app.py
```

The correct command depends on the framework used in the actual application.

---

## 🧪 Example Usage

You can use the following process to test the application.

1. Open the [live application](https://house-price-india-prediction-3.onrender.com/).
2. Enter the property's structural information.
3. Specify living area, lot area, and other available measurements.
4. Select the relevant property condition and view options.
5. Enter construction and renovation information.
6. Include available location and nearby-amenity details.
7. Submit the form using the valuation button.
8. Review the estimated property value and additional indicators.

For meaningful results, use realistic inputs that match the expected feature ranges and definitions of the training dataset.

---

## 📊 Model Evaluation

A regression model should be evaluated on data that was not used to train it.

Useful evaluation metrics for this project include:

| Metric | Description |
|---|---|
| Mean Absolute Error (MAE) | Measures the average absolute difference between actual and predicted prices |
| Mean Squared Error (MSE) | Measures the average squared prediction error |
| Root Mean Squared Error (RMSE) | Expresses the square root of the mean squared error in the target's units |
| R² Score | Measures how well the model explains variation in the target values relative to a baseline |

### Model Performance Results

The actual scores should be added after checking the model evaluation results from the training process.

| Evaluation Metric | Result |
|---|---|
| Mean Absolute Error (MAE) | Add measured result |
| Mean Squared Error (MSE) | Add measured result |
| Root Mean Squared Error (RMSE) | Add measured result |
| R² Score | Add measured result |

**Note:** Do not interpret the R² score as ordinary classification accuracy. Report metrics calculated from the actual model evaluation rather than estimated or invented values.

---

## 🌍 Potential Applications

House price prediction models can be useful in several contexts:

- **Property Research:** Obtain an initial estimate when comparing residential properties.
- **Real Estate Analytics:** Explore relationships between property characteristics and estimated prices.
- **Investment Research:** Use predicted values as one input when evaluating potential property investments.
- **Market Analysis:** Study how structural and location-related characteristics relate to property values.
- **Machine Learning Education:** Demonstrate supervised regression and the deployment of a trained model.

The application is intended as a machine learning demonstration and should not be treated as a professional property appraisal.

---

## ⚠️ Limitations

Machine learning predictions depend on the data used to train the model and the way the application processes input features.

Important considerations include:

- **Data Dependency:** Predictions reflect patterns learned from the training dataset.
- **Market Changes:** Property prices can change over time, while a trained model may not reflect recent market conditions.
- **Location Differences:** A model trained on one geographical market may not generalize reliably to another.
- **Feature Sensitivity:** Missing, inaccurate, or unrealistic inputs can affect the prediction.
- **Model Assumptions:** Linear Regression may not fully capture complex nonlinear relationships between property features and market prices.
- **Estimated Values:** A predicted price is not a guaranteed transaction price or formal property valuation.
- **Rental Estimates:** Any estimated monthly rent should be interpreted according to the application's implemented calculation and validated against relevant rental data.

Users should independently verify important property decisions using current market information and qualified professionals.

---

## 🔒 Responsible Usage

This application is designed for educational and exploratory property valuation.

- Avoid using predictions as the sole basis for financial or property investment decisions.
- Verify important information independently.
- Use inputs consistent with the model's expected features.
- Do not assume that a predicted value represents a current market offer.
- Treat supplementary valuation indicators as estimates rather than guarantees.

---

## 🚀 Future Improvements

Potential improvements include:

- Compare Linear Regression against Random Forest, Gradient Boosting, and other regression algorithms.
- Add a model evaluation dashboard with verified performance metrics.
- Improve data validation and error handling.
- Provide clearer explanations of the factors contributing to each prediction.
- Add visualizations showing relationships between property features and predicted values.
- Evaluate the model on recent and geographically relevant property data.
- Add automated tests for input processing and prediction logic.
- Improve model monitoring and retraining workflows.
- Document the dataset source, preprocessing steps, and training methodology.
- Add screenshots and example predictions to the repository documentation.

---

## 🎓 Learning Outcomes

This project demonstrates practical concepts in:

- Supervised machine learning
- Regression-based prediction
- Property feature representation
- Loading and using a saved model
- Connecting machine learning models to an application
- Developing an interactive property valuation interface
- Managing Python dependencies
- Deploying an application to a cloud hosting platform
- Documenting a machine learning project using GitHub

---

## 🔧 Troubleshooting

### 1. Dependency Installation Errors

Upgrade pip and reinstall the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation still fails, check the Python version and the package versions listed in `requirements.txt`.

### 2. Model File Not Found

Verify that `linear_model.pkl` exists in the expected location.

If the application loads the model using a relative path, run the application from the project directory or update the model path to resolve it correctly.

### 3. Prediction Errors

Check that the input features supplied to the model match the feature names, order, data types, and preprocessing expected during training.

### 4. Application Does Not Start

Check the terminal output for missing dependencies, import errors, port conflicts, or other startup problems.

### 5. Live Application Takes Time to Load

Applications hosted on free cloud services may experience startup delays after periods of inactivity. Allow some time for the service to respond and retry if necessary.

---

## 🤝 Contributing

Contributions and suggestions are welcome.

To propose an improvement:

1. Fork the repository.
2. Create a new branch.
3. Implement your changes.
4. Test the updated application.
5. Submit a pull request describing your changes.

Potential contributions include improved model evaluation, data validation, documentation, visualizations, and interface enhancements.

---

## 📜 License

A license has not been specified in this documentation.

Before distributing or reusing this project, add an appropriate license file to the repository and update this section accordingly.

---

## 👨‍💻 Author

**Siddhesh**

GitHub Repository:  
https://github.com/Siddhesh030/House-Price-Prediction

Live Application:  
https://house-price-india-prediction-3.onrender.com/

---

## ⭐ Support

If you find this project useful for learning about machine learning, regression, and web application deployment, consider giving the repository a star on GitHub.

Thank you for visiting the project!
