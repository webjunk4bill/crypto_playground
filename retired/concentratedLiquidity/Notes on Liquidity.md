# Notes on Liquidity

## Price
- Price is always in y/x.  It's important how the price moves
- x is typically ETH, therefore for most tokens, as the price goes up, the actual ratio of y/x goes down
- In the code, the absolute value of the number doesn't matter, what matters is the direction
- However, when converting from USD and how we think of price, there can be some inversions that need to be managed carefully in the code
- The upper range and lower range can change depending on which way the price is moving.  Does that mean p_a and p_b swap when the price is inverted?
- If the price is inverted, adding token x will increase the price, if not, it will decrease it
- Unless price is always determined in WETH or the token 0.  So I need to be thinking about it in the price of WETH, not BRETT or SPEC, etc.
- Therefore, Pb would always want to be the "upper range" and the "larger" value of token 'y'
- Duh, it's not the "price" in terms of USD, it's the price in terms of y/x.  So y goes up, the lp "price" goes up.  Damn...

## Ranges
- I need to create a good naming convention for the "upper" and "lower" price ranges.  
- The lower range is when the price (in terms of y/x) has fallen to the minium and all we are left with is the y token
- The upper range is when the price (in terms of y/x) has risen to the maximum and all we are left with is the x token
- Should I find the ranges in native price first and then convert to LP ranges?

## USD -> Liquidity -> USD
- There needs to be ease of movement in the code going between "usd" price and "liquidity" price
- I need a clear distinction between operating in USD and operating in liquidity prices

## Information sources
- DEX LP: here we have the "Base" and "Quote" tokens.  The base is usually the token we are interested in providing liquidity for
- Coingecko: information just on the base token
- Uniswap Liquidity: Comes in as "x" and "y" tokens, "x" is typically the quote token, but not always.  It's almost always WETH
- Maybe it is easier to create a pointer to the base/quote using token_x and y.  Then deal with x/y when in "liquidity" mode and base/quote when in usd mode
  - This doesn't seem to work :(
- The price must be inverted moving from usd to liquidity if the "base" token is the "y" token
- The hourly coingecko price is the usd price of the base token (sometimes expressed in ETH)
  - the liquidity price must be in the form of y/x
  - When calculating the seed amount of the two tokens we need to understand the usd value of each
- If I set up an object (like the Simple Token) for my position, I should be able to access it with the different named variables