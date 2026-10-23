import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Locate and load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("LinearRegression model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model from '{MODEL_PATH}': {e}")
    model = None

# Exact feature ordering expected by the model
FEATURE_KEYS = [
    'number of bedrooms',
    'number of bathrooms',
    'living area',
    'lot area',
    'number of floors',
    'waterfront present',
    'number of views',
    'condition of the house',
    'grade of the house',
    'Area of the house(excluding basement)',
    'Area of the basement',
    'Built Year',
    'Renovation Year',
    'lot_area_renov',
    'Number of schools nearby',
    'Distance from the airport'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valuora AI | Precision Real Estate Valuation Platform</title>

    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Cabinet+Grotesk:wght@700;800;900&display=swap" rel="stylesheet">

    <!-- FontAwesome & Chart.js -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --bg-base: #090d16;
            --bg-card: rgba(18, 25, 41, 0.85);
            --bg-card-hover: rgba(26, 36, 58, 0.95);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-accent: rgba(56, 189, 248, 0.35);
            --brand-primary: #38bdf8;
            --brand-secondary: #6366f1;
            --brand-emerald: #10b981;
            --text-heading: #f8fafc;
            --text-body: #94a3b8;
            --text-muted: #64748b;
            --input-bg: rgba(13, 19, 33, 0.9);
            --glow: 0 0 30px rgba(56, 189, 248, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 80%, rgba(56, 189, 248, 0.1) 0%, transparent 45%);
            color: var(--text-body);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Top Navigation Header */
        .navbar {
            border-bottom: 1px solid var(--border-subtle);
            backdrop-filter: blur(20px);
            position: sticky;
            top: 0;
            z-index: 50;
            background: rgba(9, 13, 22, 0.75);
            padding: 0.85rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }

        .brand-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
            box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
        }

        .brand-text h1 {
            font-family: 'Cabinet Grotesk', sans-serif;
            font-size: 1.35rem;
            color: var(--text-heading);
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .brand-text span {
            font-size: 0.75rem;
            color: var(--brand-primary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        .nav-actions {
            display: flex;
            align-items: center;
            gap: 1.2rem;
        }

        .currency-box {
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            border-radius: 100px;
            padding: 5px 14px;
            gap: 8px;
            font-size: 0.85rem;
            color: var(--text-heading);
        }

        .currency-box select {
            background: transparent;
            border: none;
            color: var(--brand-primary);
            font-weight: 700;
            font-size: 0.85rem;
            outline: none;
            cursor: pointer;
        }

        .currency-box select option {
            background: #121929;
            color: #fff;
        }

        .badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.72rem;
            padding: 4px 10px;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            font-weight: 600;
        }

        .badge-live .pulse-dot {
            width: 6px;
            height: 6px;
            background: #34d399;
            border-radius: 50%;
            box-shadow: 0 0 8px #34d399;
        }

        /* Container Layout */
        .wrapper {
            max-width: 1440px;
            margin: 0 auto;
            padding: 2rem;
            width: 100%;
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 2rem;
            flex: 1;
        }

        @media (max-width: 1024px) {
            .wrapper {
                grid-template-columns: 1fr;
                padding: 1.2rem;
            }
        }

        /* Left Workspace Form Panel */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 1.75rem;
            backdrop-filter: blur(16px);
            position: relative;
        }

        .card-header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .presets-box {
            display: flex;
            gap: 8px;
        }

        .btn-preset {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-subtle);
            color: var(--text-body);
            border-radius: 8px;
            padding: 5px 10px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-preset:hover {
            border-color: var(--brand-primary);
            color: var(--brand-primary);
        }

        /* Section Categorization */
        .category-legend {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--brand-primary);
            margin: 1.25rem 0 0.85rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .category-legend::after {
            content: "";
            flex: 1;
            height: 1px;
            background: var(--border-subtle);
        }

        .inputs-matrix {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.1rem;
        }

        .form-field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-field label {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-heading);
            display: flex;
            justify-content: space-between;
        }

        .form-field label small {
            color: var(--text-muted);
            font-weight: 400;
        }

        .input-control {
            background: var(--input-bg);
            border: 1px solid var(--border-subtle);
            color: #ffffff;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .input-control:focus {
            border-color: var(--brand-primary);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }

        select.input-control option {
            background: #0d1321;
        }

        .action-submit-btn {
            margin-top: 1.75rem;
            width: 100%;
            background: linear-gradient(135deg, var(--brand-primary) 0%, var(--brand-secondary) 100%);
            color: #ffffff;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            cursor: pointer;
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .action-submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(56, 189, 248, 0.4);
        }

        .action-submit-btn:active {
            transform: translateY(0);
        }

        /* Right Panel: Analytics & Output */
        .right-column {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .valuation-hero-card {
            background: linear-gradient(145deg, rgba(30, 41, 67, 0.7), rgba(15, 23, 42, 0.95));
            border: 1px solid var(--border-accent);
            border-radius: 20px;
            padding: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: var(--glow);
        }

        .valuation-hero-card::after {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 150px;
            height: 150px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.2) 0%, transparent 70%);
            pointer-events: none;
        }

        .val-tagline {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--brand-primary);
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .valuation-number {
            font-family: 'Cabinet Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            color: #ffffff;
            margin: 0.6rem 0;
            letter-spacing: -1px;
            line-height: 1.1;
        }

        /* Metric Pill Strip */
        .metric-strips {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 1.25rem;
            padding-top: 1.25rem;
            border-top: 1px solid var(--border-subtle);
        }

        .metric-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .metric-item .lbl {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        .metric-item .val {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-heading);
        }

        /* Chart Canvas Card */
        .chart-box {
            position: relative;
            height: 280px;
            width: 100%;
            margin-top: 1rem;
        }

        footer {
            text-align: center;
            padding: 1.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-subtle);
            margin-top: auto;
        }
    </style>
</head>
<body>

    <!-- Header Navigation -->
    <header class="navbar">
        <a href="#" class="brand-wrap">
            <div class="brand-icon">
                <i class="fa-solid fa-layer-group"></i>
            </div>
            <div class="brand-text">
                <h1>Valuora AI</h1>
                <span>Enterprise Valuation Engine</span>
            </div>
        </a>

        <div class="nav-actions">
            <div class="currency-box">
                <i class="fa-solid fa-coins" style="color: var(--brand-primary);"></i>
                <select id="currencySelector" onchange="convertAndDisplay()">
                    <option value="USD" data-symbol="$" data-rate="1">USD ($)</option>
                    <option value="INR" data-symbol="₹" data-rate="83.5">INR (₹)</option>
                    <option value="EUR" data-symbol="€" data-rate="0.92">EUR (€)</option>
                    <option value="GBP" data-symbol="£" data-rate="0.79">GBP (£)</option>
                    <option value="CAD" data-symbol="C$" data-rate="1.38">CAD (C$)</option>
                    <option value="AUD" data-symbol="A$" data-rate="1.52">AUD (A$)</option>
                </select>
            </div>
            <div class="badge-live">
                <span class="pulse-dot"></span> Model Ready
            </div>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="wrapper">
        <!-- Input Form Section -->
        <section class="card">
            <div class="card-header-bar">
                <div class="card-title">
                    <i class="fa-solid fa-sliders" style="color: var(--brand-primary);"></i>
                    Property Specifications
                </div>
                <div class="presets-box">
                    <button class="btn-preset" type="button" onclick="loadPreset('starter')">Suburban Starter</button>
                    <button class="btn-preset" type="button" onclick="loadPreset('luxury')">Waterfront Luxury</button>
                </div>
            </div>

            <form id="propertyForm">
                <!-- Group 1: Core Layout -->
                <div class="category-legend"><i class="fa-solid fa-house"></i> Layout & Structural Capacity</div>
                <div class="inputs-matrix">
                    <div class="form-field">
                        <label>Bedrooms <small>Count</small></label>
                        <input class="input-control" type="number" name="number of bedrooms" id="f_bedrooms" value="3" step="1" min="1" max="15" required>
                    </div>
                    <div class="form-field">
                        <label>Bathrooms <small>Count</small></label>
                        <input class="input-control" type="number" name="number of bathrooms" id="f_bathrooms" value="2.25" step="0.25" min="0.5" max="10" required>
                    </div>
                    <div class="form-field">
                        <label>Floors <small>Levels</small></label>
                        <input class="input-control" type="number" name="number of floors" id="f_floors" value="2" step="0.5" min="1" max="4" required>
                    </div>
                    <div class="form-field">
                        <label>Condition <small>Scale 1-5</small></label>
                        <select class="input-control" name="condition of the house" id="f_condition">
                            <option value="1">1 - Poor</option>
                            <option value="2">2 - Fair</option>
                            <option value="3" selected>3 - Average</option>
                            <option value="4">4 - Good</option>
                            <option value="5">5 - Exceptional</option>
                        </select>
                    </div>
                </div>

                <!-- Group 2: Dimension Metrics -->
                <div class="category-legend"><i class="fa-solid fa-ruler-combined"></i> Dimensions & Square Footage</div>
                <div class="inputs-matrix">
                    <div class="form-field">
                        <label>Living Area <small>Sq. Ft.</small></label>
                        <input class="input-control" type="number" name="living area" id="f_living" value="2150" step="10" required>
                    </div>
                    <div class="form-field">
                        <label>Total Lot Area <small>Sq. Ft.</small></label>
                        <input class="input-control" type="number" name="lot area" id="f_lot" value="6500" step="50" required>
                    </div>
                    <div class="form-field">
                        <label>Above Basement Area <small>Sq. Ft.</small></label>
                        <input class="input-control" type="number" name="Area of the house(excluding basement)" id="f_above_base" value="1650" step="10" required>
                    </div>
                    <div class="form-field">
                        <label>Basement Area <small>Sq. Ft.</small></label>
                        <input class="input-control" type="number" name="Area of the basement" id="f_basement" value="500" step="10" required>
                    </div>
                    <div class="form-field">
                        <label>Renovated Lot Area <small>Sq. Ft.</small></label>
                        <input class="input-control" type="number" name="lot_area_renov" id="f_lot_renov" value="6200" step="50" required>
                    </div>
                    <div class="form-field">
                        <label>Design Grade <small>1 (Low) - 13 (Mansion)</small></label>
                        <input class="input-control" type="number" name="grade of the house" id="f_grade" value="8" min="1" max="13" required>
                    </div>
                </div>

                <!-- Group 3: Location & Environment -->
                <div class="category-legend"><i class="fa-solid fa-map-location-dot"></i> Heritage, Views & Location</div>
                <div class="inputs-matrix">
                    <div class="form-field">
                        <label>Waterfront View</label>
                        <select class="input-control" name="waterfront present" id="f_waterfront">
                            <option value="0" selected>No (Standard)</option>
                            <option value="1">Yes (Waterfront)</option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label>View Score <small>0 - 4 Index</small></label>
                        <input class="input-control" type="number" name="number of views" id="f_views" value="1" min="0" max="4" required>
                    </div>
                    <div class="form-field">
                        <label>Year Built <small>Year</small></label>
                        <input class="input-control" type="number" name="Built Year" id="f_built" value="1998" min="1800" max="2026" required>
                    </div>
                    <div class="form-field">
                        <label>Renovation Year <small>0 if none</small></label>
                        <input class="input-control" type="number" name="Renovation Year" id="f_renov" value="0" min="0" max="2026" required>
                    </div>
                    <div class="form-field">
                        <label>Nearby Schools <small>Count</small></label>
                        <input class="input-control" type="number" name="Number of schools nearby" id="f_schools" value="3" min="0" max="10" required>
                    </div>
                    <div class="form-field">
                        <label>Airport Distance <small>Miles / Km</small></label>
                        <input class="input-control" type="number" name="Distance from the airport" id="f_airport" value="14" min="1" max="100" required>
                    </div>
                </div>

                <button type="submit" class="action-submit-btn" id="submitBtn">
                    <i class="fa-solid fa-bolt-lightning"></i> Compute Property Valuation
                </button>
            </form>
        </section>

        <!-- Right Insights Panel -->
        <section class="right-column">
            <!-- Hero Valuation Card -->
            <div class="valuation-hero-card">
                <div class="val-tagline">
                    <i class="fa-solid fa-chart-line"></i> Estimated Market Value
                </div>
                <div class="valuation-number" id="predictedPriceText">$0</div>
                <p style="font-size: 0.8rem; color: var(--text-muted);">
                    Calculated using multi-variable regression matching localized property market factors.
                </p>

                <div class="metric-strips">
                    <div class="metric-item">
                        <span class="lbl">Price / Sq.Ft.</span>
                        <span class="val" id="metricPpsf">--</span>
                    </div>
                    <div class="metric-item">
                        <span class="lbl">Est. Monthly Rent</span>
                        <span class="val" id="metricRent">--</span>
                    </div>
                    <div class="metric-item">
                        <span class="lbl">Quality Tier</span>
                        <span class="val" id="metricTier">Optimal</span>
                    </div>
                </div>
            </div>

            <!-- Visual Breakdown Chart -->
            <div class="card" style="flex: 1;">
                <div class="card-title">
                    <i class="fa-solid fa-chart-column" style="color: var(--brand-secondary);"></i>
                    Key Feature Weight Contribution
                </div>
                <div class="chart-box">
                    <canvas id="featureChart"></canvas>
                </div>
            </div>
        </section>
    </main>

    <footer>
        Valuora AI &copy; 2026 | Next-Gen Real Estate Market Intelligence
    </footer>

    <script>
        let latestRawUSD = 0;
        let chartInstance = null;

        // Presets logic
        function loadPreset(type) {
            if (type === 'starter') {
                document.getElementById('f_bedrooms').value = 3;
                document.getElementById('f_bathrooms').value = 1.75;
                document.getElementById('f_floors').value = 1;
                document.getElementById('f_condition').value = 3;
                document.getElementById('f_living').value = 1450;
                document.getElementById('f_lot').value = 4500;
                document.getElementById('f_above_base').value = 1450;
                document.getElementById('f_basement').value = 0;
                document.getElementById('f_lot_renov').value = 4500;
                document.getElementById('f_grade').value = 7;
                document.getElementById('f_waterfront').value = 0;
                document.getElementById('f_views').value = 0;
                document.getElementById('f_built').value = 1985;
                document.getElementById('f_renov').value = 0;
                document.getElementById('f_schools').value = 2;
                document.getElementById('f_airport').value = 18;
            } else if (type === 'luxury') {
                document.getElementById('f_bedrooms').value = 5;
                document.getElementById('f_bathrooms').value = 4.5;
                document.getElementById('f_floors').value = 2.5;
                document.getElementById('f_condition').value = 5;
                document.getElementById('f_living').value = 4800;
                document.getElementById('f_lot').value = 14000;
                document.getElementById('f_above_base').value = 3600;
                document.getElementById('f_basement').value = 1200;
                document.getElementById('f_lot_renov').value = 12500;
                document.getElementById('f_grade').value = 11;
                document.getElementById('f_waterfront').value = 1;
                document.getElementById('f_views').value = 3;
                document.getElementById('f_built').value = 2012;
                document.getElementById('f_renov').value = 2020;
                document.getElementById('f_schools').value = 4;
                document.getElementById('f_airport').value = 9;
            }
            triggerValuation();
        }

        // Initialize Chart
        function initChart() {
            const ctx = document.getElementById('featureChart').getContext('2d');
            chartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Living Space', 'Grade Rating', 'Lot Footprint', 'Bathrooms', 'Views'],
                    datasets: [{
                        label: 'Relative Valuation Influence',
                        data: [60, 40, 25, 20, 15],
                        backgroundColor: [
                            'rgba(56, 189, 248, 0.85)',
                            'rgba(99, 102, 241, 0.85)',
                            'rgba(16, 185, 129, 0.85)',
                            'rgba(245, 158, 11, 0.85)',
                            'rgba(236, 72, 153, 0.85)'
                        ],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            ticks: { color: '#64748b', font: { family: 'Plus Jakarta Sans', size: 11 } },
                            grid: { display: false }
                        },
                        y: {
                            ticks: { color: '#64748b' },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' }
                        }
                    }
                }
            });
        }

        // Currency Converter Display
        function convertAndDisplay() {
            if (latestRawUSD <= 0) return;

            const selector = document.getElementById('currencySelector');
            const rate = parseFloat(selector.options[selector.selectedIndex].getAttribute('data-rate'));
            const currencyCode = selector.value;
            const converted = latestRawUSD * rate;

            const livingArea = parseFloat(document.getElementById('f_living').value) || 1;
            const ppsf = converted / livingArea;
            const rent = converted * 0.0055; // Standard 0.55% monthly rental rule-of-thumb

            animateValue('predictedPriceText', converted, currencyCode);

            const symbol = selector.options[selector.selectedIndex].getAttribute('data-symbol');
            document.getElementById('metricPpsf').innerText = `${symbol}${ppsf.toFixed(0)}/sqft`;
            document.getElementById('metricRent').innerText = `${symbol}${rent.toFixed(0)}/mo`;

            const grade = parseInt(document.getElementById('f_grade').value);
            document.getElementById('metricTier').innerText = grade >= 10 ? 'High-End Luxury' : (grade >= 7 ? 'Mid-Market' : 'Entry Tier');
        }

        function animateValue(elementId, targetValue, currencyCode) {
            const el = document.getElementById(elementId);
            const formatter = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: currencyCode,
                maximumFractionDigits: 0
            });
            el.innerText = formatter.format(targetValue);
        }

        async function triggerValuation() {
            const form = document.getElementById('propertyForm');
            const formData = new FormData(form);
            const payload = {};
            formData.forEach((val, key) => payload[key] = parseFloat(val));

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    latestRawUSD = Math.max(0, data.prediction);
                    convertAndDisplay();

                    // Update Chart Dynamically based on current inputs
                    if (chartInstance) {
                        const living = payload['living area'] || 0;
                        const grade = (payload['grade of the house'] || 1) * 300;
                        const lot = (payload['lot area'] || 0) * 0.1;
                        const baths = (payload['number of bathrooms'] || 1) * 400;
                        const views = ((payload['number of views'] || 0) + 1) * 500;

                        chartInstance.data.datasets[0].data = [living, grade, lot, baths, views];
                        chartInstance.update();
                    }
                } else {
                    alert('Prediction Error: ' + data.error);
                }
            } catch (err) {
                console.error(err);
            }
        }

        document.getElementById('propertyForm').addEventListener('submit', (e) => {
            e.preventDefault();
            triggerValuation();
        });

        window.addEventListener('DOMContentLoaded', () => {
            initChart();
            triggerValuation(); // Initial load calculation
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': 'Predictive model not loaded. Check linear_model.pkl'})

    try:
        data = request.json
        feature_vector = [float(data.get(k, 0)) for k in FEATURE_KEYS]
        input_array = np.array([feature_vector])
        prediction = model.predict(input_array)[0]

        return jsonify({
            'success': True,
            'prediction': float(prediction)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Binds to Render's dynamic PORT variable or defaults to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
