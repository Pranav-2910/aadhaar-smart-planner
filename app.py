import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="UIDAI Smart Biometric Planner & Audit System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium glassmorphic dark-theme CSS
st.markdown("""
    <style>
        /* Base styles */
        .reportview-container {
            background: #0e1117;
        }
        .main {
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Inter', sans-serif;
        }
        
        /* Metric card container */
        .metric-card {
            background: linear-gradient(135deg, rgba(31, 41, 55, 0.7) 0%, rgba(17, 24, 39, 0.7) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 6px 24px 0 rgba(0, 0, 0, 0.3);
        }
        
        /* Custom titles and headers */
        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.025em;
        }
        .gradient-text {
            background: -webkit-linear-gradient(45deg, #4f46e5, #06b6d4, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        /* Badges */
        .badge {
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .badge-low { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-medium { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-high { background-color: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
        
        /* Simulator box style */
        .sim-box {
            background: rgba(255, 255, 255, 0.02);
            border: 1px dashed rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
        }
        
        /* Footer */
        .footer {
            font-size: 0.85rem;
            text-align: center;
            color: #8b949e;
            margin-top: 50px;
            padding: 20px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Helper functions to load data and models using simple relative paths
@st.cache_resource
def load_assets():
    # Load data files directly from data/ folder
    district_df = pd.read_csv("data/district_features.csv")
    district_df['date'] = pd.to_datetime(district_df['date'])
    pincode_df = pd.read_csv("data/pincode_hotspots.csv")
    
    # Load models directly from models/ folder
    forecaster = joblib.load("models/demand_forecaster.pkl")
    anomaly_detector = joblib.load("models/anomaly_detector.pkl")
    category_mappings = joblib.load("models/category_mappings.pkl")
    
    return district_df, pincode_df, forecaster, anomaly_detector, category_mappings

# Try loading assets
try:
    district_df, pincode_df, forecaster, anomaly_detector, category_mappings = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading assets from folder. Details: {e}")

if assets_loaded:
    # Sidebar logo/title
    st.sidebar.markdown('## 🔮 UIDAI Smart Planner')
    st.sidebar.markdown('**Strategic Biometric Management System**')
    st.sidebar.markdown('---')
    
    # Filters
    st.sidebar.subheader("Select Region")
    states_list = sorted(district_df['state'].unique())
    selected_state = st.sidebar.selectbox("State", states_list)
    
    districts_list = sorted(district_df[district_df['state'] == selected_state]['district'].unique())
    selected_district = st.sidebar.selectbox("District", districts_list)
    
    # Date Range of dataset
    min_date = district_df['date'].min()
    max_date = district_df['date'].max()
    
    st.sidebar.subheader("Target Date Selection")
    forecast_limit_date = max_date + timedelta(days=14)
    selected_date = st.sidebar.date_input(
        "Select Date (for forecast/audit)",
        value=max_date,
        min_value=min_date,
        max_value=forecast_limit_date
    )
    selected_date_dt = pd.to_datetime(selected_date)
    
    # Title / Header
    st.markdown('<h1 class="gradient-text">UIDAI Smart Biometric Planner & Audit System</h1>', unsafe_allow_html=True)
    st.markdown('A data-driven machine learning system to predict infrastructure loads, optimize ASK resource allocation, and detect operator anomalies.')
    st.markdown('---')
    
    # Real-time National Operations Status Bar
    national_day_data = district_df[district_df['date'] == selected_date_dt]
    if len(national_day_data) > 0:
        total_national_volume = national_day_data['total_bio'].sum()
        total_national_anomalies = national_day_data['is_anomaly_z'].sum()
    else:
        total_national_volume = district_df[district_df['date'] == max_date]['total_bio'].sum()
        total_national_anomalies = 0

    status_color = "#10b981" if total_national_anomalies == 0 else "#f59e0b" if total_national_anomalies < 3 else "#ef4444"
    status_text = "🟢 STABLE" if total_national_anomalies == 0 else "🟡 CAUTION" if total_national_anomalies < 3 else "🔴 CRITICAL AUDIT ALERT"
    
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 20px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
            <div>
                <span style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;">SYSTEM HEALTH STATUS</span>
                <div style="font-size: 1.1rem; font-weight: 700; color: {status_color};">{status_text}</div>
            </div>
            <div>
                <span style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;">NATIONAL TRANSACTION VOLUME</span>
                <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">{total_national_volume:,} updates</div>
            </div>
            <div>
                <span style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;">ACTIVE SECURITY ALERTS</span>
                <div style="font-size: 1.1rem; font-weight: 700; color: {status_color if total_national_anomalies == 0 else '#ef4444'};">{total_national_anomalies} Districts Flagged</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Demand Forecasting & Planning", 
        "📅 12-Month Strategic Planner",
        "🎛️ 'What-If' Resource Simulator",
        "🛡️ Security & Operations Audit"
    ])
    
    # 1. Autoregressive Prediction Engine Setup
    hist_district = district_df[
        (district_df['state'] == selected_state) & 
        (district_df['district'] == selected_district)
    ].sort_values('date').reset_index(drop=True)
    
    # Perform forecasting logic
    is_hist = False
    predicted_val = 0
    risk_label = 'Low'
    cap_ratio = 0.0
    bio_5_17 = 0
    bio_17 = 0
    
    if selected_date_dt <= max_date:
        day_data = hist_district[hist_district['date'] == selected_date_dt]
        if len(day_data) > 0:
            predicted_val = day_data['total_bio'].values[0]
            risk_label = day_data['risk_level'].values[0]
            cap_ratio = day_data['capacity_ratio'].values[0]
            bio_5_17 = day_data['bio_age_5_17'].values[0]
            bio_17 = day_data['bio_age_17_'].values[0]
            is_hist = True
            
    # If the date is in the future OR has no historical record for that day, run the prediction model
    if not is_hist:
        latest_hist = hist_district.iloc[-1] if len(hist_district) > 0 else None
        
        if latest_hist is not None:
            lag_7_date = selected_date_dt - timedelta(days=7)
            lag_14_date = selected_date_dt - timedelta(days=14)
            
            def get_lag_val(d):
                match = hist_district[hist_district['date'] == d]
                return match['total_bio'].values[0] if len(match) > 0 else latest_hist['total_bio']
                
            lag_7_val = get_lag_val(lag_7_date)
            lag_14_val = get_lag_val(lag_14_date)
            
            roll_mean_30 = latest_hist['total_bio_roll_mean_30']
            roll_mean_7 = latest_hist['total_bio_roll_mean_7']
            roll_std_30 = latest_hist['total_bio_roll_std_30']
            
            features_dict = {
                'state': [selected_state],
                'district': [selected_district],
                'total_bio_lag_7': [lag_7_val],
                'total_bio_lag_14': [lag_14_val],
                'total_bio_roll_mean_7': [roll_mean_7],
                'total_bio_roll_mean_30': [roll_mean_30],
                'total_bio_roll_std_30': [roll_std_30],
                'day': [selected_date_dt.day],
                'month': [selected_date_dt.month],
                'weekday': [selected_date_dt.weekday()]
            }
            X_pred = pd.DataFrame(features_dict)
            X_pred['state'] = pd.Categorical(X_pred['state'], categories=category_mappings['state'])
            X_pred['district'] = pd.Categorical(X_pred['district'], categories=category_mappings['district'])
            
            try:
                if hasattr(forecaster, "predict"):
                    predicted_val = int(forecaster.predict(X_pred)[0])
                else:
                    X_pred_rf = X_pred.copy()
                    X_pred_rf['state'] = X_pred_rf['state'].cat.codes
                    X_pred_rf['district'] = X_pred_rf['district'].cat.codes
                    predicted_val = int(forecaster.predict(X_pred_rf)[0])
            except Exception as e:
                predicted_val = int(latest_hist['total_bio'])
                
            predicted_val = max(0, predicted_val)
            cap_ratio = predicted_val / (roll_mean_30 + 1)
            
            if cap_ratio <= 1.0:
                risk_label = 'Low'
            elif cap_ratio <= 1.3:
                risk_label = 'Medium'
            else:
                risk_label = 'High'
            
            age_5_17_ratio = latest_hist['bio_age_5_17'] / (latest_hist['total_bio'] + 1)
            bio_5_17 = int(predicted_val * age_5_17_ratio)
            bio_17 = int(predicted_val * (1 - age_5_17_ratio))

    # ------------------ TAB 1: Demand Forecasting & Planning ------------------
    with tab1:
        st.subheader("📊 Infrastructure Capacity Planning")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Predicted Demand</span>
                    <h2 style="margin: 8px 0; color: #ffffff;">{predicted_val:,}</h2>
                    <span style="font-size: 0.8rem; color: #10b981;">{"(Historical)" if is_hist else "(Forecasted)"} updates/day</span>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            badge_class = f"badge-{risk_label.lower()}"
            st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Infrastructure Load Risk</span>
                    <div style="margin: 8px 0;"><span class="badge {badge_class}" style="font-size: 1.4rem;">{risk_label}</span></div>
                    <span style="font-size: 0.8rem; color: #8b949e;">Ratio: {cap_ratio:.2f}x of standard load</span>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            if risk_label == 'Low':
                recom = "Standard Operations"
                recom_color = "#10b981"
                recom_desc = "No Action Required"
            elif risk_label == 'Medium':
                recom = "Deploy 1 Mobile ASK Kit"
                recom_color = "#f59e0b"
                recom_desc = "Monitor for surges"
            else:
                recom = "Deploy 2+ Mobile Kits"
                recom_color = "#ef4444"
                recom_desc = "Open temporary ASK camps"
                
            st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Kit Recommendation</span>
                    <h3 style="margin: 8px 0; color: {recom_color};">{recom}</h3>
                    <span style="font-size: 0.8rem; color: #8b949e;">{recom_desc}</span>
                </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Age Segmentation</span>
                    <h4 style="margin: 8px 0; color: #ffffff;">5-17 yrs: <span style="color:#06b6d4;">{bio_5_17:,}</span></h4>
                    <h4 style="margin: 0; color: #ffffff;">17+ yrs: <span style="color:#4f46e5;">{bio_17:,}</span></h4>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Demand Forecast Graph
        st.subheader("📈 60-Day Historic Load vs. 14-Day Model Forecast")
        
        graph_data = hist_district.tail(60).copy()
        
        future_rows = []
        latest_hist_val = hist_district.iloc[-1]
        for i in range(1, 15):
            f_date = max_date + timedelta(days=i)
            f_lag_7 = f_date - timedelta(days=7)
            f_lag_14 = f_date - timedelta(days=14)
            
            def get_roll_lag(d):
                m_hist = hist_district[hist_district['date'] == d]
                if len(m_hist) > 0:
                    return m_hist['total_bio'].values[0]
                for row in future_rows:
                    if row['date'] == f_date:
                        return row['total_bio']
                return latest_hist_val['total_bio']
                
            l_7 = get_roll_lag(f_lag_7)
            l_14 = get_roll_lag(f_lag_14)
            
            f_feat = pd.DataFrame({
                'state': [selected_state],
                'district': [selected_district],
                'total_bio_lag_7': [l_7],
                'total_bio_lag_14': [l_14],
                'total_bio_roll_mean_7': [latest_hist_val['total_bio_roll_mean_7']],
                'total_bio_roll_mean_30': [latest_hist_val['total_bio_roll_mean_30']],
                'total_bio_roll_std_30': [latest_hist_val['total_bio_roll_std_30']],
                'day': [f_date.day],
                'month': [f_date.month],
                'weekday': [f_date.weekday()]
            })
            f_feat['state'] = pd.Categorical(f_feat['state'], categories=category_mappings['state'])
            f_feat['district'] = pd.Categorical(f_feat['district'], categories=category_mappings['district'])
            
            try:
                if hasattr(forecaster, "predict"):
                    f_pred = int(forecaster.predict(f_feat)[0])
                else:
                    f_feat_rf = f_feat.copy()
                    f_feat_rf['state'] = f_feat_rf['state'].cat.codes
                    f_feat_rf['district'] = f_feat_rf['district'].cat.codes
                    f_pred = int(forecaster.predict(f_feat_rf)[0])
            except:
                f_pred = int(latest_hist_val['total_bio'])
                
            f_pred = max(0, f_pred)
            future_rows.append({
                'date': f_date,
                'state': selected_state,
                'district': selected_district,
                'total_bio': f_pred,
                'type': 'Forecast'
            })
            
        future_df = pd.DataFrame(future_rows)
        graph_data['type'] = 'Historical'
        
        plot_df = pd.concat([graph_data[['date', 'total_bio', 'type']], future_df[['date', 'total_bio', 'type']]])
        
        fig = go.Figure()
        hist_plot = plot_df[plot_df['type'] == 'Historical']
        fig.add_trace(go.Scatter(
            x=hist_plot['date'], y=hist_plot['total_bio'],
            mode='lines+markers', name='Historical actuals',
            line=dict(color='#06b6d4', width=3),
            marker=dict(size=4)
        ))
        
        fore_plot = plot_df[plot_df['type'] == 'Forecast']
        fore_plot = pd.concat([hist_plot.tail(1), fore_plot])
        fig.add_trace(go.Scatter(
            x=fore_plot['date'], y=fore_plot['total_bio'],
            mode='lines+markers', name='14-Day Forecast',
            line=dict(color='#ef4444', width=3, dash='dash'),
            marker=dict(size=5, symbol='diamond')
        ))
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            height=320,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Biometric Transactions"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig)
        
        # Download predictions CSV button
        download_df = future_df[['date', 'state', 'district', 'total_bio']].copy()
        download_df['date'] = download_df['date'].dt.strftime('%Y-%m-%d')
        download_df = download_df.rename(columns={'total_bio': 'forecasted_biometric_updates'})
        csv_data = download_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download 14-Day Forecast Data as CSV",
            data=csv_data,
            file_name=f"{selected_district}_14day_forecast.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Drilldown Treemap & Hotspots
        col_left, col_right = st.columns([1.1, 0.9])
        
        with col_left:
            st.subheader("📁 Hierarchical Contribution Map")
            st.write("Visual breakdown of the selected State's updates by District (Hierarchical Treemap):")
            
            state_data = district_df[district_df['state'] == selected_state].groupby(['state', 'district'])['total_bio'].sum().reset_index()
            
            treemap_fig = px.treemap(
                state_data,
                path=['state', 'district'],
                values='total_bio',
                color='total_bio',
                color_continuous_scale='Blues',
                labels={'total_bio': 'Total Updates'}
            )
            treemap_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=5, r=5, t=10, b=5),
                height=320
            )
            st.plotly_chart(treemap_fig)
            
        with col_right:
            st.subheader("🎯 Hyper-local Pincode Hotspots")
            st.write(f"Top pincodes driving biometric traffic in **{selected_district}**:")
            dist_pincodes = pincode_df[
                (pincode_df['state'] == selected_state) & 
                (pincode_df['district'] == selected_district)
            ].sort_values(by='total_updates', ascending=False).reset_index(drop=True)
            
            if len(dist_pincodes) > 0:
                dist_pincodes['percentage'] = (dist_pincodes['total_updates'] / dist_pincodes['total_updates'].sum()) * 100
                st.dataframe(
                    dist_pincodes[['pincode', 'total_updates', 'percentage']].head(6).style.format({
                        'total_updates': '{:,}',
                        'percentage': '{:.1f}%'
                    }),
                    width="stretch"
                )
            else:
                st.info("No pincode details available.")
                
    # ------------------ TAB 2: 12-Month Strategic Planner ------------------
    with tab2:
        st.subheader("📅 12-Month Strategic Capacity Planner (2026 Projection)")
        st.write("Projecting the macro-demand profile for the entire year of 2026 based on the 2025 baseline and customizable annual growth assumptions.")
        
        col_g1, col_g2 = st.columns([1.5, 2.5])
        
        with col_g1:
            st.markdown("### 📈 Strategic Assumptions")
            growth_rate = st.slider("Assumed Annual Growth Rate (%)", -20, 50, 5) / 100.0
            
            # Aggregate 2025 monthly data for the selected district
            monthly_data = hist_district.groupby('month')['total_bio'].sum().reset_index()
            
            # Handle missing Jan/Feb in 2025 dataset (approximate using March/average)
            all_months = pd.DataFrame({'month': range(1, 13)})
            monthly_data = pd.merge(all_months, monthly_data, on='month', how='left')
            
            march_val = monthly_data[monthly_data['month'] == 3]['total_bio'].values[0]
            if pd.isna(march_val):
                march_val = monthly_data['total_bio'].mean()
            
            monthly_data['total_bio'] = monthly_data['total_bio'].fillna(march_val)
            
            # Project 2026 values
            monthly_data['projected_2026'] = (monthly_data['total_bio'] * (1 + growth_rate)).astype(int)
            
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            monthly_data['month_name'] = monthly_data['month'].map(month_names)
            
            total_projected_2026 = monthly_data['projected_2026'].sum()
            peak_month = monthly_data.loc[monthly_data['projected_2026'].idxmax(), 'month_name']
            peak_val = monthly_data['projected_2026'].max()
            
            st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 15px;">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Total Forecasted updates (2026)</span>
                    <h2 style="margin: 8px 0; color: #ffffff;">{total_projected_2026:,}</h2>
                    <span style="font-size: 0.8rem; color: #10b981;">For district: {selected_district}</span>
                </div>
                
                <div class="metric-card">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Peak Capacity demand month</span>
                    <h3 style="margin: 8px 0; color: #ef4444;">{peak_month} <span style="font-size: 1rem; color: #8b949e;">({peak_val:,} updates)</span></h3>
                    <span style="font-size: 0.8rem; color: #8b949e;">Procurement target threshold</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Dynamic Strategic Action Plan
            if total_projected_2026 > 100000:
                infra_rec = "🏛️ Upgrade to Permanent Aadhaar Seva Kendra (ASK) Category-A advised due to high annual load."
            elif total_projected_2026 > 30000:
                infra_rec = "🏢 Standard ASK Center Category-B is sufficient to handle baseline load."
            else:
                infra_rec = "🚐 Mobile-first strategy recommended. Maintain 1 permanent kit and rely on mobile camps."
                
            buffer_vans = max(1, int(peak_val / 5000))
            
            st.markdown(f"""
                <div class="metric-card" style="margin-top: 15px; border-left: 4px solid #4f46e5;">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase; font-weight: bold;">📋 Strategic Action Plan</span>
                    <div style="font-size: 0.85rem; color: #c9d1d9; margin-top: 8px; line-height: 1.4;">
                        <strong>Infrastructure:</strong> {infra_rec}<br><br>
                        <strong>Peak Buffer Planning:</strong> Maintain a minimum reserve of <strong>{buffer_vans} mobile enrollment vans</strong> during <strong>{peak_month}</strong> to absorb the seasonal peak surge.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with col_g2:
            st.markdown("### 📊 2026 Projected Monthly Volume")
            
            bar_fig = px.bar(
                monthly_data,
                x='month_name',
                y='projected_2026',
                color='projected_2026',
                color_continuous_scale='blues',
                labels={'projected_2026': 'Projected Volume', 'month_name': 'Month'}
            )
            bar_fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=10, b=20),
                height=320,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Updates")
            )
            st.plotly_chart(bar_fig)

    # ------------------ TAB 3: 'What-If' Resource Simulator ------------------
    with tab3:
        st.subheader("🎛️ Operational 'What-If' Planner & Capacity Optimizer")
        st.write("Simulate changes in local processing capacity (staffing, machines, and vans) to see if you can handle the forecasted demand without causing citizen delay bottleneck surges.")
        
        col_s1, col_s2 = st.columns([1, 1.2])
        
        with col_s1:
            st.markdown("### ⚙️ Simulator Variables")
            st.write("Configure resource allocations for the selected day:")
            
            sim_centers = st.slider("Permanent Aadhaar Centers in District", 1, 20, 5)
            sim_kits = st.slider("Enrollment Kits/Counters per Center", 1, 10, 2)
            sim_capacity = st.number_input("Average Processing Capacity per Kit (Updates/Day)", 10, 100, 40)
            sim_vans = st.slider("Mobile Enrollment Vans Deployed", 0, 10, 1)
            sim_van_cap = st.number_input("Capacity per Mobile Van (Updates/Day)", 50, 200, 100)
            
        with col_s2:
            st.markdown("### 📊 Capacity & Congestion Output")
            
            perm_capacity = sim_centers * sim_kits * sim_capacity
            van_capacity = sim_vans * sim_van_cap
            total_capacity = perm_capacity + van_capacity
            
            congestion_index = predicted_val / (total_capacity + 1)
            
            if congestion_index <= 0.8:
                congest_status = "Optimal Load (Under Capacity)"
                status_color = "#10b981"
                status_desc = "Resource allocation is fully sufficient. Citizens will face minimal wait times."
            elif congestion_index <= 1.1:
                congest_status = "Moderate Load (Near Limit)"
                status_color = "#f59e0b"
                status_desc = "Resource capacity is close to limit. Slight queues may develop during peak hours."
            else:
                congest_status = "Overloaded (Capacity Deficit)"
                status_color = "#ef4444"
                status_desc = "DEMAND EXCEEDS CAPACITY! Citizens will face severe delays. Resource additions strongly advised."
                
            st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 20px;">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Total Local Capacity</span>
                    <h2 style="margin: 8px 0; color: #ffffff;">{total_capacity:,} <span style="font-size: 1rem; color: #8b949e;">updates/day</span></h2>
                    <span style="font-size: 0.85rem; color: #8b949e;">Permanent: {perm_capacity:,} | Mobile Vans: {van_capacity:,}</span>
                </div>
                
                <div class="metric-card">
                    <span style="font-size: 0.9rem; color: #8b949e; text-transform: uppercase;">Congestion Index</span>
                    <h2 style="margin: 8px 0; color: {status_color};">{congestion_index:.2f}</h2>
                    <h4 style="margin: 4px 0; color: #ffffff;">{congest_status}</h4>
                    <p style="font-size: 0.85rem; color: #8b949e; margin-bottom: 0;">{status_desc}</p>
                </div>
            """, unsafe_allow_html=True)
            
            if congestion_index > 1.0:
                deficit = predicted_val - total_capacity
                vans_needed = int(np.ceil(deficit / sim_van_cap))
                st.markdown(f"""
                    <div class="sim-box">
                        <strong style="color: #ef4444;">⚠️ Optimization Recommendation:</strong><br>
                        Deploy at least <strong>{vans_needed} additional</strong> mobile enrollment van(s) or add 
                        <strong>{int(np.ceil(deficit / sim_capacity))}</strong> counter kits to permanent centers to balance demand.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="sim-box">
                        <strong style="color: #10b981;">✅ Safe Zone:</strong><br>
                        No additional resource deployment is required. Capacity covers demand with a surplus of <strong>{total_capacity - predicted_val:,}</strong> updates.
                    </div>
                """, unsafe_allow_html=True)

    # ------------------ TAB 4: Security & Operations Audit ------------------
    with tab4:
        st.subheader("🛡️ Suspicious Operation Spikes & Anomaly Audits")
        st.write("Using rolling statistical analysis (Z-Score) and unsupervised Isolation Forest modeling to find transaction volume anomalies. Spikes can indicate unauthorized 'pop-up' update camps or compromised systems.")
        
        district_anomalies = district_df[
            (district_df['state'] == selected_state) & 
            (district_df['is_anomaly_z'] == 1)
        ].sort_values(by='z_score', ascending=False).reset_index(drop=True)
        
        col_a, col_b = st.columns([1.2, 1.8])
        
        with col_a:
            st.markdown(f"### 🚨 Anomalies in **{selected_state}**")
            st.write("Districts sorted by statistical deviation strength (Z-Score):")
            
            if len(district_anomalies) > 0:
                st.dataframe(
                    district_anomalies[['date', 'district', 'total_bio', 'z_score']].head(10).style.format({
                        'total_bio': '{:,}',
                        'z_score': '{:.2f}'
                    }),
                    width="stretch"
                )
            else:
                st.success(f"No major anomalies detected for districts in {selected_state}!")
                
        with col_b:
            st.markdown("### ⚠️ National Operational Risk Map (Simulated Audit Alert)")
            
            national_anomalies = district_df[
                (district_df['date'] == selected_date_dt) & 
                (district_df['is_anomaly_z'] == 1)
            ].sort_values('z_score', ascending=False)
            
            if len(national_anomalies) > 0:
                st.warning(f"On {selected_date_dt.strftime('%d-%B-%Y')}, the model flagged {len(national_anomalies)} national operational anomalies:")
                st.dataframe(
                    national_anomalies[['state', 'district', 'total_bio', 'z_score']].style.format({
                        'total_bio': '{:,}',
                        'z_score': '{:.2f}'
                    }),
                    width="stretch"
                )
            else:
                st.success(f"All districts nationwide report stable/normal operations on {selected_date_dt.strftime('%d-%B-%Y')}!")
                
        st.markdown("---")
        st.subheader("💡 Audit Methodology Notes")
        st.markdown("""
        * **Z-Score Anomaly Engine:** Flags instances where a district's daily volume is **> 3.0 standard deviations** above its 30-day moving average. Useful for detecting sudden local events or localized operator spikes.
        * **Why it matters:** In real-world Aadhaar deployments, finding sudden spikes helps UIDAI compliance and audit teams flag operator accounts that might be doing unauthorized registrations, system abuse, or using unregistered GPS coordinate overrides.
        """)

    # Footer
    st.markdown("""
        <div class="footer">
            UIDAI Hackathon 2026 | Built for Real-Life Strategic Infrastructure Planning
        </div>
    """, unsafe_allow_html=True)
