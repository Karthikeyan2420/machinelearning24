# Current price of 24 carat gold per gram (you can update this value)
price_per_gram_24_carat = 7664.0  # Example price in your currency

# Calculate the purity of 8 carat and 10 carat gold
purity_8_carat = 8 / 24  # 8 carat gold purity
purity_10_carat = 10 / 24  # 10 carat gold purity

# Calculate the price for 8 carat and 10 carat gold
price_8_carat = purity_8_carat * price_per_gram_24_carat
price_10_carat = purity_10_carat * price_per_gram_24_carat

# Display the prices
print(f"Current price of 8 carat gold per gram: {price_8_carat:.2f}")
print(f"Current price of 10 carat gold per gram: {price_10_carat:.2f}")