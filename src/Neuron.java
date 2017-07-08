/**
 * Created by dubangshi on 2017/7/2.
 */
import java.util.*;

public class Neuron {
    static int counter = 0;
    final public int id;  // auto increment, starts at 0
    Connection biasConnection;
    final double bias = -1;
    double output;

    ArrayList<Connection> Inconnections = new ArrayList<Connection>();
    HashMap<Integer,Connection> connectionLookup = new HashMap<Integer,Connection>(); //key：from神经元 ；value：与this 和 key神经元 相关的connection对象

    public Neuron(){
        id = counter;
        counter++;
    }

    /**
     * Compute Sj = Wij*Aij + w0j*bias
     */
    public void calculateOutput(){
        double temps = 0;
        for(Connection con : Inconnections){
            Neuron leftNeuron = con.getFromNeuron();
            double weight = con.getWeight();
            double a = leftNeuron.getOutput(); //output from previous layer

            temps += (weight * a);
        }
        temps += (biasConnection.getWeight()*bias);

        output = g(temps);
    }


    double g(double x) {
        return sigmoid(x);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 +  (Math.exp(-x)));
    }

    public void addInConnectionsS(ArrayList<Neuron> inNeurons){
        for(Neuron n: inNeurons){
            Connection con = new Connection(n,this); //n:from神经元; this：to神经元
            Inconnections.add(con);
            connectionLookup.put(n.id, con);
        }
    }

    public Connection getConnection(int neuronIndex){
        return connectionLookup.get(neuronIndex);
    }

    public void addInConnection(Connection con){
        Inconnections.add(con);
    }
    public void addBiasConnection(Neuron n){ //此时的n为biasUnit，将单个偏移神经元与this的connection 也添加到"入连接"集合中
        Connection con = new Connection(n,this);
        biasConnection = con;
        Inconnections.add(con);
    }
    public ArrayList<Connection> getAllInConnections(){
        return Inconnections;
    }

    public double getBias() {
        return bias;
    }
    public double getOutput() {
        return output;
    }
    public void setOutput(double o){
        output = o;
    }
}