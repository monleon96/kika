NJOY_INPUT_TEMPLATE = """\
moder
 20 -25
reconr
 -25 -21
'{title}'/
 {mat} 0 0
 0.001 0 0.01 5e-08
 0 /
broadr
 -25 -21 -22
 {mat} 1 0 0 0
 0.001 1e+06 0.01 5e-08
 {T}
 0 /
moder
 -22 30
heatr
 -25 -22 -21 /
 {mat} 4 0 0 0 0 /
 302 402 442 444 /
heatr
 -25 -22 -23 /
 {mat} 5 0 1 0 2 /
 302 303 402 442 444 /
thermr
 0 -21 -22 /
 0 {mat} 16 1 1 0 0 1 221 2 /
 {T}
 0.001 5.0
gaspr
 -25 -22 -21  /
acer
 -25 -21 0 40 41
 1 0 1 {suff} /
'{title}'/
 {mat} {T}
 1 1 1
 /
acer
 0 40 42 40 41
 7 1 1 -1 /
'{title}'/
viewr
 42 43
stop
"""