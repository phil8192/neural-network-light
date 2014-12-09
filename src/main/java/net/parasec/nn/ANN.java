package net.parasec.nn;

import java.util.Random;


/**
 * feed-forward neural network.
 * note that this (matrix based) implementation is based on some of my old code 
 * from 2005. it is mathematically sound, however the implementation is quite
 * poor.
 */
public final class ANN {
  private final static Logger LOG = Logger.getLogger(ANN.class);


  // network weight matrix:
  // [layer][neuron][weight]
  // where each layer contains neurons with incomming weights.
  private double[][][] weights; 

  // neuron outputs. output of each neuron after presenting a single instance.
  private final double[][] outputs;

  // neuron errors.
  private final double[][] deltas;

  // previous weight change.
  // needed for momentum calculation.
  private double[][][] preDW;

  // network structure
  private final int[] structure;
  private final int nLayers, nOutputs, nInputs;

  // prng.
  private final Random prng; 


  public ANN(final double min, final double max, final int[] structure, 
      final Random prng) {
    this.structure = structure;
    this.prng = prng;
    nLayers = structure.length-1;
    nOutputs = structure[structure.length-1];
    nInputs = structure[0];
    outputs = new double[structure.length][];
    LOG.info("initialising network with structure: " + 
        java.util.Arrays.toString(structure));
    for(int i = outputs.length; --i >= 0; )
      outputs[i] = new double[structure[i]];
    deltas = new double[structure.length-1][];
    for(int i = deltas.length; --i >= 0; )
      deltas[i] = new double[structure[i+1]];
    weights = new double[nLayers][][];
    for(int i = nLayers; --i >= 0; )
      weights[i] = new double[structure[i+1]][structure[i]+1];
    for(int i = weights.length; --i >= 0; )
      for(int j = weights[i].length; --j >= 0; )
        for(int k = weights[i][j].length; --k >= 0; )
          weights[i][j][k] = getRandomWeight(min, max);
    preDW = new double[nLayers][][];
    for(int i = nLayers; --i >= 0; )
      preDW[i] = new double[structure[i+1]][structure[i]+1];
  }
 
  public double[] feedForward(final double[] instance) {
    // initialise 1st outputs to be inputs,
    // since input neurons have no activation function
    outputs[0] = instance;
    // for each layer i
    for(int i = 1; i < outputs.length; i++) {
      // for each neuron j in this layer
      for(int j =0; j <outputs[i].length; j++) {
        // bias weight (1) first.
        double sum = weights[i-1][j][0];
        // for each input to this neuron
        for(int k = 1; k < weights[i-1][j].length; k++) {
          sum += weights[i-1][j][k]*outputs[i-1][k-1];
        }
        outputs[i][j] = sigmoid(sum);
      }
    }
    return outputs[nLayers];
  }

  public void backPropagateError(final double[] desiredOutput, 
      final double learningRate, final double momentum) {
    calculateError(desiredOutput);
    // for each layer
    for(int i = weights.length; --i >= 0; ) {
      // for each neuron
      for(int j = weights[i].length; --j >= 0; ) {
        // bias weight (1) first.
        final double dwBias 
            = (learningRate*deltas[i][j])+(momentum*preDW[i][j][0]);
        weights[i][j][0] += dwBias;
        preDW[i][j][0] = dwBias;
        // for each weight connected to this neuron
        for(int k = weights[i][j].length; --k > 0; ) {
          final double dw = (learningRate*(deltas[i][j]*outputs[i][k-1]))
              + (momentum*preDW[i][j][k]);
          weights[i][j][k] += dw;
          preDW[i][j][k] = dw;
        }
      }
    }
  }

  public double[][][] getWeights() {
    return weights;
  }

  public double[][][] cloneWeights() {
    final double[][][] networkWeights = new double[nLayers][][];
    for(int i = nLayers; --i>= 0; )
      networkWeights[i] = new double[structure[i+1]][structure[i]+1];
      for(int i = networkWeights.length; --i>= 0; )
        for(int j = networkWeights[i].length; --j>= 0; )
          for(int k = networkWeights[i][j].length; --k >= 0; )
            networkWeights[i][j][k] = weights[i][j][k];
    return networkWeights;
  }

  public void initialiseWeights(final double[][][] weights) {
    this.weights = weights;
    preDW = new double[nLayers][][];
    for(int i = nLayers; --i >= 0; )
      preDW[i] = new double[structure[i+1]][structure[i]+1];    
  }

  // 2nd hotspot.
  private void calculateError(final double[] desiredOutput) {
    // output layer
    for(int i = nOutputs; --i >=0; ) {
      final double sd = sigmoidDerivative(outputs[nLayers][i]);
      deltas[nLayers-1][i] = sd*(desiredOutput[i]-outputs[nLayers][i]);
    }
    // for each hidden layer
    for(int i = nLayers-1; --i >= 0; ) {
      for(int j = deltas[i].length; --j >= 0; ) {
        double sum = 0;
        // for each outgoing weight to neurons in layer +1, sum
        // outgoingWeight*delta(layer +1)
        for(int k = deltas[i+1].length; --k >= 0; )
          sum += weights[i+1][k][j+1]*deltas[i+1][k];
        deltas[i][j] = sigmoidDerivative(outputs[i+1][j]) * sum;
      }
    }
  }

  //
  //
  // note: sigmoid(double) self time = 42.61%.
  // sigmoid calculation is the main hotspot; this is due to the expensive
  // exp function. perhaps the accuracy is not necessary. take a look at
  // http://code.google.com/p/fastapprox/ (approximate and vectorised versions 
  // of functions commonly used in machine learning).
  // 
  // this, from: seems good.
  // martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
  //
  //
  private double fastExp1(final double x) {
    final long tmp = (long) (1512775 * x + 1072632447);
    return Double.longBitsToDouble(tmp << 32);
  }

  // 1st hotspot. see note^.
  private double sigmoid(final double x) {
    //return 1/(1+Math.exp(-x));
    return 1/(1+fastExp1(-x));
  }

  private double sigmoidDerivative(final double sOutput) {
    return sOutput*(1-sOutput);
  }

  private double tanh(final double x) {
    return Math.tanh(x);
  }

  private double tanhDerivative(final double thOutput) {
    return 1-(thOutput*thOutput);
  }

  private double getRandomWeight(final double min, final double max) {
    return prng.nextDouble()*(max-min)+min;
  }

}

