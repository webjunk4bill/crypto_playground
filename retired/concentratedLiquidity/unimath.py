import math

min_tick = -887272
max_tick = 887272

q96 = 2 ** 96
eth = 10 ** 18


def price_to_tick(p):
    return math.floor(math.log(p, 1.0001))


def price_to_sqrtp(p):
    return int(math.sqrt(p) * q96)


def sqrtp_to_price(sqrtp):
    return (sqrtp / q96) ** 2


def tick_to_sqrtp(t):
    return int((1.0001 ** (t / 2)) * q96)


def liquidity0(amount, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return (amount * (pa * pb) / q96) / (pb - pa)


def liquidity_x(amount_x, current_price, upper_range):
    """
    The "x" token is the token that is in the denominator of the price function y/x.
    Normally this would be the "base" token because "y" would be the "backing" or "quote" token.
    However, that doesn't always align with how the uniswap ticks are read from the LP contract.
    One needs to be careful to properly set x and y
    :param amount_x: Token amount in standard terms (not WEI)
    :param current_price: Current price of x in y/x
    :param upper_range: Price where all x token is gone
    :return: liquidity
    """
    amount = amount_x * eth
    pa = price_to_sqrtp(current_price)
    pb = price_to_sqrtp(upper_range)
    return liquidity0(amount, pa, pb)


def liquidity1(amount, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return amount * q96 / (pb - pa)


def liquidity_y(amount_y, current_price, lower_range):
    """
    The "y" token is the token that is in the numerator of the price function y/x.
    Normally this would be the "quote" token
    However, that doesn't always align with how the uniswap ticks are read from the LP contract.
    One needs to be careful to properly set x and y
    :param amount_y: Token amount in standard terms (not WEI)
    :param current_price: Current price of x in y/x
    :param lower_range: Price where all y token is gone
    :return:
    """
    amount = amount_y * eth
    pa = price_to_sqrtp(current_price)
    pb = price_to_sqrtp(lower_range)
    return liquidity1(amount, pa, pb)


def calc_amount0(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return int(liq * q96 * (pb - pa) / pb / pa)


def calc_amount_x(liquidity, current_price, upper_range):
    pa = price_to_sqrtp(current_price)
    pb = price_to_sqrtp(upper_range)
    return calc_amount0(liquidity, pa, pb) / eth


def calc_amount1(liq, pa, pb):
    if pa > pb:
        pa, pb = pb, pa
    return int(liq * (pb - pa) / q96)


def calc_amount_y(liquidity, current_price, lower_range):
    pa = price_to_sqrtp(current_price)
    pb = price_to_sqrtp(lower_range)
    return calc_amount1(liquidity, pa, pb) / eth


def calc_upper_range_pb(x, y, lower_range, current_price):
    p = current_price ** 0.5
    pa = lower_range ** 0.5
    return ((p * y) / (pa * p * x - current_price * x + y)) ** 2


def calc_lower_range_pa(x, y, upper_range, current_price):
    p = current_price ** 0.5
    pb = upper_range ** 0.5
    return (y / pb / x + p - y / p / x) ** 2


"""
Notes:
Need to read this over and over: https://uniswapv3book.com/milestone_1/calculating-liquidity.html
calc_amount 0 refers to the "x" token or the denominator in the pricing function
calc amount 1 refers to the "y" token or the numerator
In the Brett/ETH LP for example, the price is around 0.00005 eth/brett or 20000 brett/eth
If you are using 0.00005 (eth/brett) than x = brett and y is eth
If you are using 20000 (brett/eth) than x = eth and y is brett
Remember when using brett/eth and increase in "price" is actually a decrease in usd (less eth/brett)
When putting in an unequal ratio, I can just use that ratio to "pre-swap" the tokens and get close.  Just putting in
the single sided liquidity doesn't come out with the right answer
"""

'''
Examples:
# Liquidity provision
price_low = 4545
price_cur = 5000
price_upp = 5500

print(f"Price range: {price_low}-{price_upp}; current price: {price_cur}")

sqrtp_low = price_to_sqrtp(price_low)
sqrtp_cur = price_to_sqrtp(price_cur)
sqrtp_upp = price_to_sqrtp(price_upp)

amount_eth = 1 * eth
amount_usdc = 5000 * eth

liq0 = liquidity0(amount_eth, sqrtp_cur, sqrtp_upp)
liq1 = liquidity1(amount_usdc, sqrtp_cur, sqrtp_low)
liq = int(min(liq0, liq1))

print(f"Deposit: {amount_eth/eth} ETH, {amount_usdc/eth} USDC; liquidity: {liq}")

# Swap USDC for ETH
amount_in = 42 * eth

print(f"\nSelling {amount_in/eth} USDC")

price_diff = (amount_in * q96) // liq
price_next = sqrtp_cur + price_diff

print("New price:", (price_next / q96) ** 2)
print("New sqrtP:", price_next)
print("New tick:", price_to_tick((price_next / q96) ** 2))

amount_in = calc_amount1(liq, price_next, sqrtp_cur)
amount_out = calc_amount0(liq, price_next, sqrtp_cur)

print("USDC in:", amount_in / eth)
print("ETH out:", amount_out / eth)

# Swap ETH for USDC
amount_in = 0.01337 * eth

print(f"\nSelling {amount_in/eth} ETH")

price_next = int((liq * q96 * sqrtp_cur) // (liq * q96 + amount_in * sqrtp_cur))

print("New price:", (price_next / q96) ** 2)
print("New sqrtP:", price_next)
print("New tick:", price_to_tick((price_next / q96) ** 2))

amount_in = calc_amount0(liq, price_next, sqrtp_cur)
amount_out = calc_amount1(liq, price_next, sqrtp_cur)

print("ETH in:", amount_in / eth)
print("USDC out:", amount_out / eth)
'''
