package com.nopeya.neural.perceptron;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.*;

import java.util.Arrays;

/**
 * 逻辑与感知机
 * @author: Nopeya
 * @Date: 2020/10/31 10:07
 */
public class AndPerceptronNoLearn extends NeuralNetwork {

    public AndPerceptronNoLearn(int inputNeuronsCount) {
        this.createNetwork(inputNeuronsCount);
    }

    private void createNetwork(int inputNeuronsCount) {

        // 网络类型
        this.setNetworkType(NeuralNetworkType.PERCEPTRON);

        // 输入层
        NeuronProperties inputNeuronProperties = new NeuronProperties();
        inputNeuronProperties.setProperty("neuronType", InputNeuron.class);
        Layer inputLayer = LayerFactory.createLayer(inputNeuronsCount, inputNeuronProperties);
        this.addLayer(inputLayer);
        // 偏置神经元使用贝叶斯神经元
        inputLayer.addNeuron(new BiasNeuron());

        // 传输函数, 使用阶梯函数
        NeuronProperties outputNeuronProperties = new NeuronProperties();
        outputNeuronProperties.setProperty("transferFunction", TransferFunctionType.STEP);
        Layer outputLayer = LayerFactory.createLayer(1, outputNeuronProperties);
        this.addLayer(outputLayer);

        // 全连接
        ConnectionFactory.fullConnect(inputLayer, outputLayer);
        NeuralNetworkFactory.setDefaultIO(this);
        // 连接权重
        Neuron outputNeuron = outputLayer.getNeuronAt(0);
        outputNeuron.getInputConnections()[0].getWeight().setValue(1);      // 输入权重
        outputNeuron.getInputConnections()[1].getWeight().setValue(1);      // 输入权重
        outputNeuron.getInputConnections()[2].getWeight().setValue(-1.5);   // 神经元偏置
    }

    public static void main(String[] args) {
        DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{Double.NaN}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{Double.NaN}));


        AndPerceptronNoLearn perceptron = new AndPerceptronNoLearn(2);
        for (DataSetRow row : trainingSet.getRows()) {
            perceptron.setInput(row.getInput());
            perceptron.calculate();
            double[] output = perceptron.getOutput();
            System.out.println(Arrays.toString(row.getInput()) + " = " + Arrays.toString(output));

        }

    }

}
