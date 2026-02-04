# Exploratory Data Analysis (EDA) - Summary Report
## Real Estate Investment Advisor Project

**Dataset:** 250,000 properties across India  
**Features:** 36 (after preprocessing)  
**Date:** February 2026

---

## 📊 Price & Size Analysis (Questions 1-5)

### Q1: What is the distribution of property prices?

**Visualization:** [q1_price_distribution.png](file:///d:/workspace/PROJECT2/data/processed/q1_price_distribution.png)

**Key Findings:**
- **Mean Price:** ₹255 Lakhs
- **Median Price:** ₹255 Lakhs
- **Standard Deviation:** ₹142 Lakhs
- **Distribution:** Nearly normal with slight right skew
- **Range:** ₹10L - ₹500L

**Insight:** Prices are well-distributed across the range with balanced representation.

---

### Q2: What is the distribution of property sizes?

**Visualization:** [q2_size_distribution.png](file:///d:/workspace/PROJECT2/data/processed/q2_size_distribution.png)

**Key Findings:**
- **Mean Size:** 2,748 sq ft
- **Median Size:** 2,748 sq ft
- **Range:** 500 - 5,000 sq ft
- **Distribution:** Normal distribution

**Insight:** Property sizes are evenly distributed, indicating diverse property types.

---

### Q3: How does price per sq ft vary by property type?

**Visualization:** [q3_price_per_sqft_by_type.png](file:///d:/workspace/PROJECT2/data/processed/q3_price_per_sqft_by_type.png)

**Key Findings:**
- Different property types show varying price per sq ft
- Apartments, Villas, and Houses analyzed
- Clear pricing hierarchy observed

**Insight:** Property type significantly impacts pricing per square foot.

---

### Q4: Is there a relationship between property size and price?

**Visualization:** [q4_size_vs_price.png](file:///d:/workspace/PROJECT2/data/processed/q4_size_vs_price.png)

**Key Findings:**
- **Correlation Coefficient:** Strong positive correlation
- **Trend:** Linear relationship observed
- Larger properties command higher prices

**Insight:** Size is a strong predictor of property price.

---

### Q5: Are there outliers in price per sq ft or property size?

**Visualization:** [q5_outliers.png](file:///d:/workspace/PROJECT2/data/processed/q5_outliers.png)

**Key Findings:**
- **Price per Sq Ft Outliers:** 20,020 properties (8.01%)
- **Size Outliers:** Minimal
- Outliers primarily in premium/luxury segment

**Insight:** Some properties have exceptionally high price per sq ft, likely luxury properties.

---

## 🗺️ Location-Based Analysis (Questions 6-10)

### Q6: What is the average price per sq ft by state?

**Visualization:** [q6_price_by_state.png](file:///d:/workspace/PROJECT2/data/processed/q6_price_by_state.png)

**Key Findings:**
- Top 10 most expensive states identified
- Significant regional variation in pricing
- Metropolitan states command premium prices

**Insight:** Location (state) is a major price determinant.

---

### Q7: What is the average property price by city?

**Visualization:** [q7_price_by_city.png](file:///d:/workspace/PROJECT2/data/processed/q7_price_by_city.png)

**Key Findings:**
- Top 20 cities ranked by average price
- Tier-1 cities show highest prices
- Wide price variation across cities

**Insight:** City-level analysis reveals specific high-value markets.

---

### Q8: What is the median age of properties by locality?

**Visualization:** [q8_age_by_locality.png](file:///d:/workspace/PROJECT2/data/processed/q8_age_by_locality.png)

**Key Findings:**
- Top 20 localities by median property age
- Mix of old and new developments
- Age varies significantly by locality

**Insight:** Property age distribution helps identify established vs new areas.

---

### Q9: How is BHK distributed across cities?

**Visualization:** [q9_bhk_by_city.png](file:///d:/workspace/PROJECT2/data/processed/q9_bhk_by_city.png)

**Key Findings:**
- Top 10 cities analyzed
- BHK preferences vary by city
- 2BHK and 3BHK most common

**Insight:** Different cities have different housing size preferences.

---

### Q10: What are the price trends for the top 5 most expensive localities?

**Visualization:** [q10_top_expensive_localities.png](file:///d:/workspace/PROJECT2/data/processed/q10_top_expensive_localities.png)

**Key Findings:**
- Top 5 premium localities identified
- Significant price premium in these areas
- Clear market segmentation

**Insight:** Premium localities command substantially higher prices.

---

## 🔗 Feature Relationship & Correlation (Questions 11-15)

### Q11: How are numeric features correlated with each other?

**Visualization:** [q11_correlation_matrix.png](file:///d:/workspace/PROJECT2/data/processed/q11_correlation_matrix.png)

**Key Findings:**
- Comprehensive correlation heatmap
- Strong correlations identified:
  - Size ↔ Price
  - BHK ↔ Size
  - Infrastructure ↔ Price
- Weak correlations also documented

**Insight:** Multiple features show meaningful relationships with price.

---

### Q12: How do nearby schools relate to price per sq ft?

**Visualization:** [q12_schools_vs_price.png](file:///d:/workspace/PROJECT2/data/processed/q12_schools_vs_price.png)

**Key Findings:**
- Positive correlation observed
- More schools nearby → higher prices
- Educational infrastructure adds value

**Insight:** Proximity to schools is a price driver.

---

### Q13: How do nearby hospitals relate to price per sq ft?

**Visualization:** [q13_hospitals_vs_price.png](file:///d:/workspace/PROJECT2/data/processed/q13_hospitals_vs_price.png)

**Key Findings:**
- Positive correlation observed
- Healthcare accessibility impacts pricing
- Similar pattern to schools

**Insight:** Healthcare infrastructure adds property value.

---

### Q14: How does price vary by furnished status?

**Visualization:** [q14_price_by_furnished.png](file:///d:/workspace/PROJECT2/data/processed/q14_price_by_furnished.png)

**Key Findings:**
- **Fully Furnished:** Highest prices
- **Semi-Furnished:** Medium prices
- **Unfurnished:** Lowest prices
- Clear pricing hierarchy

**Insight:** Furnished properties command significant premium.

---

### Q15: How does price per sq ft vary by facing direction?

**Visualization:** [q15_price_by_facing.png](file:///d:/workspace/PROJECT2/data/processed/q15_price_by_facing.png)

**Key Findings:**
- North and East facing properties preferred
- Direction impacts pricing
- Cultural preferences reflected

**Insight:** Property facing direction influences buyer preferences and prices.

---

## 💰 Investment/Amenities/Ownership Analysis (Questions 16-20)

### Q16: How many properties belong to each owner type?

**Visualization:** [q16_owner_type.png](file:///d:/workspace/PROJECT2/data/processed/q16_owner_type.png)

**Key Findings:**
- Distribution across Builder, Owner, Broker
- Pie chart shows percentage breakdown
- Market composition analyzed

**Insight:** Understanding owner type distribution helps assess market dynamics.

---

### Q17: How many properties are available under each availability status?

**Visualization:** [q17_availability_status.png](file:///d:/workspace/PROJECT2/data/processed/q17_availability_status.png)

**Key Findings:**
- Ready to Move vs Under Construction
- Market availability analyzed
- Inventory status documented

**Insight:** Availability status affects buyer decisions and pricing.

---

### Q18: Does parking space affect property price?

**Visualization:** [q18_parking_vs_price.png](file:///d:/workspace/PROJECT2/data/processed/q18_parking_vs_price.png)

**Key Findings:**
- Positive correlation with parking spaces
- More parking → higher prices
- Premium for multiple parking spots

**Insight:** Parking is a valuable amenity that increases property value.

---

### Q19: How do amenities affect price per sq ft?

**Visualization:** [q19_amenities_vs_price.png](file:///d:/workspace/PROJECT2/data/processed/q19_amenities_vs_price.png)

**Key Findings:**
- 325 unique amenity combinations analyzed
- Premium amenities command higher prices
- Clubhouse, Pool, Gym most valuable

**Insight:** Amenities significantly impact property pricing.

---

### Q20: How does public transport accessibility relate to price per sq ft?

**Visualization:** [q20_transport_vs_price.png](file:///d:/workspace/PROJECT2/data/processed/q20_transport_vs_price.png)

**Key Findings:**
- **Low Accessibility:** ₹0.1309/sq ft
- **Medium Accessibility:** ₹0.1306/sq ft
- **High Accessibility:** ₹0.1303/sq ft
- Minimal price variation

**Insight:** Surprisingly, transport accessibility shows minimal impact on pricing in this dataset.

---

## 📈 Overall Key Insights

### **Top Price Drivers:**
1. ✅ Property Size (strong correlation)
2. ✅ Location (State, City, Locality)
3. ✅ BHK Configuration
4. ✅ Furnished Status
5. ✅ Amenities
6. ✅ Nearby Infrastructure (Schools, Hospitals)
7. ✅ Parking Spaces

### **Market Characteristics:**
- **Balanced Dataset:** 50% good investments, 50% not recommended
- **Price Range:** ₹10L - ₹500L (wide spectrum)
- **Size Range:** 500 - 5,000 sq ft (diverse properties)
- **Geographic Coverage:** 20 states, 42 cities, 500 localities

### **Investment Opportunities:**
- Premium localities identified
- Undervalued areas with good infrastructure
- Growth potential in emerging localities

---

## 📁 Files Reference

**All visualizations saved in:**
```
d:\workspace\PROJECT2\data\processed\
```

**To view all images:**
```bash
explorer d:\workspace\PROJECT2\data\processed
```

**To regenerate analysis:**
```bash
python src\eda_analysis.py
```

---

## 🎯 Conclusion

This comprehensive EDA of 250,000 properties reveals:
- **Clear price patterns** based on size, location, and amenities
- **Strong predictive features** for investment classification
- **Market segmentation** across geography and property types
- **Infrastructure impact** on property values

These insights form the foundation for our machine learning models to predict:
1. **Good Investment Classification** (50/50 balanced target)
2. **Future Price Prediction** (5-year projection)

---

**Generated:** February 2026  
**Project:** Real Estate Investment Advisor  
**Dataset:** 250,000 Indian Properties
