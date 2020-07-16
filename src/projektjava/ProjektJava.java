/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package projektjava;

import java.util.*;


import javax.swing.*;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
 
import org.math.plot.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
 

import java.util.Random;

import weka.core.Instances;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.SparseInstance;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.gui.explorer.ClustererPanel;
import weka.gui.visualize.*;
import weka.core.*;
 
public class ProjektJava extends JFrame {
    private JTextArea covariance_text_area[];
    private JTextArea mean_text_area[];
    private JButton add_button;
    private JButton reset_button;
    private JButton get_clusters_button;
    
    private RandomGenerator point_random;
    private double[] means = {0, 0};
    private double[][] covariances = {{0, 0}, {0, 0}};
    JTextArea num_samples_text_area;
    private double[][] samples = new double[1000000][2];
    private MultivariateNormalDistribution distribution;
    private int num_gaussians = 0;
    private int num_samples = 50;
    private int num_samples_prev = 0;
    private Plot3DPanel plot;
    private JFrame before_clustering_frame;
        
    private class ButtonHandling implements ActionListener{
        
        private JFrame ref_window;

        ButtonHandling(JFrame window){
            ref_window = window;
        }
        
        private void create_gaussian(){
            try { // generate a new gaussian if covaiance is not a zero matrix
                num_samples = Integer.valueOf(num_samples_text_area.getText());
                if(num_samples+num_samples_prev > 1000000) {
                    JOptionPane.showMessageDialog(ref_window, "Samples' limit has been exceeded.");
                } else {                   
                    point_random = new JDKRandomGenerator();
                    // Get values of means and covariance matrix from user input in text areas
                    means[0] = Double.valueOf(mean_text_area[0].getText());
                    means[1] = Double.valueOf(mean_text_area[1].getText());
                    covariances[0][0] = Double.valueOf(covariance_text_area[0].getText());
                    covariances[0][1] = Double.valueOf(covariance_text_area[1].getText());
                    covariances[1][0] = Double.valueOf(covariance_text_area[2].getText());
                    covariances[1][1] = Double.valueOf(covariance_text_area[3].getText());
                    distribution = new MultivariateNormalDistribution(point_random, means, covariances); // create a new distribution given means and covariance matrix
                    num_gaussians += 1;
                    // generate samples from distribution
                    for(int i=num_samples_prev*num_gaussians; i<num_samples*(num_gaussians+1); i++){
                        samples[i] = distribution.sample();
                    }
                    num_samples_prev = num_samples;
                    // clear text areas after retrieving the input
                    for(int i=0; i<=1; i++){
                        mean_text_area[i].setText("0");
                    }
                    for(int i=0; i<=3; i++){
                        covariance_text_area[i].setText("0");
                    }
                    // clear plot and fill it with updated samples
                    before_clustering_frame.getContentPane().removeAll();
                    before_clustering_frame.repaint();
                    plot = new Plot3DPanel();
                    plot.addHistogramPlot("Mieszanina Gaussowska", Arrays.copyOfRange(samples, 0, num_samples*(num_gaussians+1)), 100, 100);
                    before_clustering_frame.add(plot);
                    before_clustering_frame.revalidate();
                    before_clustering_frame.repaint();
                    SwingUtilities.updateComponentTreeUI(ref_window);
                }
            } catch (Exception e) { // if
                JOptionPane.showMessageDialog(ref_window, "Covariance matrix cannot be a zero matrix.");
            }
        }
        
        private void get_clusters() throws Exception {
            // Assign each sumple to a cluster
            if(num_gaussians > 0) { // if there are no samples, there is no need to do clustering
                FastVector atts = new FastVector();
                List<Instance> instances = new ArrayList<Instance>();
                int numDimensions = samples[0].length;
                int numInstances = num_samples*(num_gaussians+1);
                for(int dim = 0; dim < numDimensions; dim++) {
                    Attribute current = new Attribute("Attribute" + dim, dim);
                    if(dim == 0) {
                        for(int obj = 0; obj < numInstances; obj++) {
                            instances.add(new SparseInstance(numDimensions));
                        }
                    }

                    for(int obj = 0; obj < numInstances; obj++) {
                        instances.get(obj).setValue(current, samples[obj][dim]);
                    }

                    atts.addElement(current);
                }

                Instances data = new Instances("Dataset", atts, instances.size());
                for(Instance inst : instances){
                    data.add(inst);
                }

                String[] options = new String[]{"-N", String.valueOf(num_gaussians)};
                EM model = new EM();
                model.setOptions(options);
                // build the clusterer
                model.buildClusterer(data);
                System.out.println(model);
                ClusterEvaluation eval = new ClusterEvaluation();
                eval.setClusterer(model);
                eval.evaluateClusterer(data);
                double logLikelihood = eval.crossValidateModel(model, data, 10, new Random(1));

                PlotData2D predData = ClustererPanel.setUpVisualizableInstances(data, eval);
                String name = "GMM";
                String cname = model.getClass().getName();
                if (cname.startsWith("weka.clusterers."))
                    name += cname.substring("weka.clusterers.".length());
                else
                    name += cname;

                VisualizePanel vp = new VisualizePanel();
                vp.setName(name + " (" + data.relationName() + ")");
                predData.setPlotName(name + " (" + data.relationName() + ")");
                vp.addPlot(predData);
                String plotName = vp.getName();
                final javax.swing.JFrame jf = 
                    new javax.swing.JFrame("Weka Clusterer Visualize: " + plotName);
                jf.setSize(500,400);
                jf.getContentPane().setLayout(new BorderLayout());
                jf.getContentPane().add(vp, BorderLayout.CENTER);
                jf.addWindowListener(new java.awt.event.WindowAdapter() {
                    public void windowClosing(java.awt.event.WindowEvent e) {
                        jf.dispose();
                    }
                });
                jf.setVisible(true);
                System.out.println(logLikelihood);
            } else {
                JOptionPane.showMessageDialog(ref_window, "Add some samples first.");
            }
        }
        
        private void clear_data() {
            num_gaussians = 0;
            before_clustering_frame.getContentPane().removeAll();
            before_clustering_frame.repaint();
            plot = new Plot3DPanel();
            before_clustering_frame.add(plot);
            before_clustering_frame.revalidate();
            before_clustering_frame.repaint();
            SwingUtilities.updateComponentTreeUI(ref_window);
        }

        public void actionPerformed(ActionEvent e) {
            JButton bt = (JButton)e.getSource();
            if(bt == add_button){
                create_gaussian();
                
            } else if(bt == reset_button) {
                clear_data();
            } else if(bt == get_clusters_button){
                try {
                    get_clusters();
                } catch (Exception ex) {
                    Logger.getLogger(ProjektJava.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            
            
        }
        
    }
    
    public ProjektJava(){
        super("Gaussian Mixture Clustering");
        // GUI for adding 2D gaussians with given means and variances
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 800);
        
        JLabel num_samples_text = new JLabel("No. of samples");
        num_samples_text_area = new JTextArea("50");
        num_samples_text_area.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        get_clusters_button = new JButton("get clusters");
        get_clusters_button.addActionListener(new ButtonHandling(this));
        
        mean_text_area = new JTextArea[2];
        for(int i=0; i<=1; i++){
            mean_text_area[i] = new JTextArea("0");
            mean_text_area[i].setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        }
        covariance_text_area = new JTextArea[4];
        for(int i=0; i<=3; i++){
            covariance_text_area[i] = new JTextArea("0");
            covariance_text_area[i].setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        }
        
        add_button = new JButton("add");
        add_button.addActionListener(new ButtonHandling(this));
        
        reset_button = new JButton("reset");
        reset_button.addActionListener(new ButtonHandling(this));
        
        get_clusters_button = new JButton("get clusters");
        get_clusters_button.addActionListener(new ButtonHandling(this));
        
        
        
        GridLayout meanGridLayout = new GridLayout(1, 2); 
        GridLayout covarianceGridLayout = new GridLayout(2, 2);
        JPanel meanPanel = new JPanel(meanGridLayout);
        JPanel covariancePanel = new JPanel(covarianceGridLayout);
        for(int i=0; i<=1; i++){
            meanPanel.add(mean_text_area[i]);
        }
        for(int i=0; i<=3; i++){
            covariancePanel.add(covariance_text_area[i]);
        }
        
        JPanel topPanel = new JPanel(new FlowLayout());
        topPanel.add(num_samples_text);
        topPanel.add(num_samples_text_area);
        JLabel meanText = new JLabel("Mean");
        topPanel.add(meanText);
        topPanel.add(meanPanel);
        JLabel covarianceText = new JLabel("Covariance");
        topPanel.add(covarianceText);
        topPanel.add(covariancePanel);
        topPanel.add(add_button);
        topPanel.add(reset_button);
        topPanel.add(get_clusters_button);

        plot = new Plot3DPanel();
        
        before_clustering_frame = new JFrame("before clustering");
        before_clustering_frame.setContentPane(plot);
        before_clustering_frame.setSize(600, 600);
        
        Container content = getContentPane();
        content.setLayout(new BorderLayout());
        content.add(topPanel, BorderLayout.NORTH);
        content.add(new JScrollPane(plot),BorderLayout.CENTER);

        
        setVisible(true);

    }

        
        
    public static void main(String[] args) throws Exception{
        /**/
        new ProjektJava();
 
        }
}