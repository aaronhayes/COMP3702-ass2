Function Feedforward returns the output layer
inputs: the input layer

let y be the output layer
let f() be the outputFunction

/ * Clear values of nodes in each layer */
for each layer l in the network do
    for each node n in l do
        l[n] = 0;

/* Feed input values to hidden layer 1 */
for each node n in input layer x do
    for each node h in hidden layer 1 hl do
        hl[h] += weighH1[h][n] * x[n]

/* Calculate value */
for each node n in hidden layer 1 hl do
    hl[n] = f(hl[n])

/* Feed hidden layer 1 values to hidden layer 2 */
for each node n in hidden layer 1 x do
    for each node h in hidden layer 2 hl do
        hl[h] += weighH2[h][n] * x[n]

/* Calculate value */
for each node n in hidden layer 2 hl do
    hl[n] = f(hl[n])


/* Feed hidden layer 2 values to output layer */
for each node n in hidden layer 2 x do
    for each node h in y do
        y[h] += weighOut[h][n] * x[n]

/* Calculate output value */
for each node n in y do
    y[n] = f(y[n] + bias[n])

return y