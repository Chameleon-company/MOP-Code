# Real-Time Melbourne Public Transport & Weather API Setup Guide

## Overview
Your MPT chatbot now supports real-time data from official Melbourne sources! This guide will help you set up the APIs to get live transport and weather information.

## 🌤️ Weather API Setup

### Option 1: Bureau of Meteorology (BOM) - FREE & Official
The BOM API is already configured and will work without any API key. It provides official Australian weather data.

**No setup required** - it's already working!

### Option 2: OpenWeather API - Fallback (Optional)
If you want a backup weather source:

1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Get your API key
4. Replace `YOUR_OPENWEATHER_API_KEY` in `index.html` with your actual key

## 🚂 Public Transport Victoria (PTV) API Setup

### Step 1: Get PTV API Key
1. Visit [PTV Timetable API](https://www.ptv.vic.gov.au/footer/data-and-reporting/datasets/ptv-timetable-api/)
2. Click "Register for API access"
3. Fill out the registration form
4. You'll receive your API key via email

### Step 2: Configure Your API Key
1. Open `index.html` in a text editor
2. Find this line:
   ```javascript
   const PTV_API_KEY = 'YOUR_PTV_API_KEY';
   ```
3. Replace `YOUR_PTV_API_KEY` with your actual PTV API key

### Step 3: Test the Integration
1. Open your webpage
2. Check the Live Status section - it should show real service status
3. Try the Journey Planner with real Melbourne stations
4. Use Real-time Departures for actual departure times

## 🎯 What You'll Get with Real APIs

### Real-Time Transport Data:
- ✅ Live service status (Good Service/Delays/Suspensions)
- ✅ Actual service disruptions and alerts
- ✅ Real departure times from PTV
- ✅ Accurate journey planning with transfers
- ✅ Current Melbourne weather conditions

### Fallback System:
- 🔄 If APIs fail, the system automatically uses realistic simulated data
- 🔄 No interruption to user experience
- 🔄 Graceful degradation ensures the chatbot always works

## 🚀 Features Now Working with Real Data

### 1. Live Status Updates
- **Trains**: Real PTV disruption data
- **Buses**: Live bus service status
- **Trams**: Current tram service information

### 2. Service Alerts
- Real disruption messages from PTV
- Live updates every 2 minutes
- Official service change notifications

### 3. Weather Widget
- Official BOM (Bureau of Meteorology) data
- Real-time temperature, humidity, wind
- Updates every 5 minutes

### 4. Journey Planner
- Real PTV route planning
- Actual transfer information
- Live departure times
- Accurate travel durations

### 5. Real-Time Departures
- Live departure boards
- Actual platform information
- Real-time delays and cancellations

## 🔧 Troubleshooting

### If APIs Don't Work:
1. Check your internet connection
2. Verify your PTV API key is correct
3. Check browser console for error messages
4. The system will automatically fall back to simulated data

### Common Issues:
- **CORS Errors**: The BOM API might have CORS restrictions. The system handles this gracefully.
- **API Limits**: PTV has rate limits. The system includes delays to respect these limits.
- **Network Issues**: All functions have fallback mechanisms.

## 📱 Testing Your Setup

### Test Real Weather:
1. Open the webpage
2. Look at the Weather & Accessibility section
3. Check if temperature and conditions are current

### Test Real Transport:
1. Click "Journey Planner"
2. Enter "Flinders Street Station" to "Melbourne Central"
3. Check if results show realistic times and routes

### Test Live Departures:
1. Click "Real-time Departures"
2. Enter "Flinders Street Station"
3. Verify departure times are current

## 🎉 You're All Set!

Once you've added your PTV API key, your MPT chatbot will provide:
- **Real-time Melbourne weather** from official sources
- **Live public transport data** from PTV
- **Accurate journey planning** with actual routes
- **Current service alerts** and disruptions
- **Live departure information** for all stations

The system is designed to work seamlessly with or without API keys, ensuring your users always get a great experience!
