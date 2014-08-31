function train returns an error value
inputs: the network, the input values (x), the desired output values (d), learning rate.

let y be the output layer
let error be the error value
let f() be the outputFunction.
let f'() be the derivative of f().

feedforward(x);

/* Calculate errors */
For each node n in the y do
    Error[n] = (y[n] - d[n]) * f'(y[n])
    error += (y[n] - d[n]) ^ 2
    
for each node n in the second hidden layer hl do
    for each node j in the output layer do
        Error_h2 += Error[j] * weightH2[j][n];
    Error_h2[n] = f'(hl[n])
    
for each node n in the first hidden layer hl do
    for each node j in the second hidden layer do
        Error_h1 += Error_h2[j] * weightH1[j][n];
    Error_h1[n] = f'(hl[n])

/* Change weights according to error */
for each node n in y do
    for each node h in second hidden layer hl do
        weight[n][h] += Error[n] * hl[h] * learning rate
      
for each node n in second hidden layer hl do
    for each node h in first hidden layer hf do
        weight[n][h] += Error_h2[n] * hf[h] * learning rate

for each node n in first hidden layer hl do
    for each node h in x hl do
        weight[n][h] += Error_h1[n] * x[h] * learning rate        

return square root of (error / size of y)