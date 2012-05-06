package org.myorg;
	
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
//import org.apache.hadoop.mapred.*;
////////////////////////////////////
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
///////////////////////////////////
import org.apache.hadoop.util.*;

import org.apache.commons.logging.*;

public class ThirdPass {

  // Logging
  static Log mapLog = LogFactory.getLog(Map.class);
  static Log reduceLog = LogFactory.getLog(Reduce.class);  

  // g^2 = m is optimal
  static long computeG(long val) {

      long estG = (long)(Math.pow((double)val, 1.5));

      List<Integer> factors  = new ArrayList<Integer>();
      for(int i=1; i <= val/2; i++)
      {
          if(val % i == 0)
          {
              factors.add(i);
          }
      }

      return factors.get(factors.size() - 1);
  }

  private static final long m = 4;
  private static final long g = computeG(m);
  
  // compute filter parameters for netid jsh263
  private static final double fromNetID = 0.362;
  private static final double desiredDensity = 0.59;
  private static final double wMin = 0.4 * fromNetID;
  private static final double wLimit = wMin + desiredDensity;

  public static class Map extends Mapper<LongWritable, Text, LongWritable, Text> { 
      
      private static long countEmissions = 0;
      
      public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        
        LongWritable one = new LongWritable(1);
        
        String nodeStr = value.toString();
        
        String[] node = nodeStr.split("\\s+");
        
        long groupNum = Long.parseLong(node[0]);
        long nodeNum = Long.parseLong(node[1]);
        long nodeLabel = Long.parseLong(node[2]);
        
        //mapLog.info("Group Number: " + groupNum);
        //mapLog.info("Node Number: " + nodeNum);
        //mapLog.info("Node Label: " + nodeLabel);
        //mapLog.info("Limit last row: " + m*(m - 1));
        long gm = g*m;
        //mapLog.info("g*m: " + gm);
        long modulus = nodeNum % gm;
        //mapLog.info("modulus: " + modulus);
        
        context.write(new LongWritable(groupNum), new Text(node[1] + " " + node[2]));
        
        //mapLog.info("------------------------------------");
        
        return;
      }
  }
  
  public static class Reduce extends Reducer<LongWritable,Text,LongWritable,LongWritable> {
    
    public class Node{
      public long groupNum;
      public long nodeNum;
      public long nodeLabel;
      public boolean visited;
      
      public Node(long gn, long nn, long nl){
        groupNum = gn;
        nodeNum = nn;
        nodeLabel = nl;
        visited = false;
      }
    }
    
    private static void dfs(long currentLabel, long newLabel,
      HashMap<Long, ArrayList<Node>> positionsMap, HashMap<Long, ArrayList<Node>> labelsMap){
        Long key = new Long(currentLabel);
        ArrayList<Node> nodes = labelsMap.get(key);
        //if(nodes.get(0).visited){
        //    return;
        //}
        
        reduceLog.info("DFS");
        
        // Iterate through each position in the labels list, running dfs
        // on each of those labels
        for(int i = 0; i < nodes.size(); i++){
            nodes.get(i).nodeLabel = newLabel;
            nodes.get(i).visited = true;
            reduceLog.info("Node " + nodes.get(i).nodeNum + " marked as visited with label " + nodes.get(i).nodeLabel);
        }
        
         for(int i = 0; i < nodes.size(); i++){
            Long nodeNum = new Long(nodes.get(i).nodeNum);
            reduceLog.info("Position: " + nodeNum.longValue());
            ArrayList<Node> nodesWithNodeNum = positionsMap.get(nodeNum);
            reduceLog.info("Number of elements at this position: " + nodesWithNodeNum.size());
            for(int j = 0; j < nodesWithNodeNum.size(); j++){
                if(!(nodesWithNodeNum.get(j).visited)){
                    reduceLog.info("Also processing label: " + nodesWithNodeNum.get(i).nodeLabel);
                    dfs(nodesWithNodeNum.get(j).nodeLabel, newLabel, positionsMap, labelsMap);
                }
            }
        }
    }
    
    public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      
      // Hashmaps with an appropriate initial capacity
      HashMap<Long, ArrayList<Node>> positionsMap = new HashMap<Long, ArrayList<Node>>((int)(m*m/g));
      HashMap<Long, ArrayList<Node>> labelsMap = new HashMap<Long, ArrayList<Node>>((int)(m*m/g));
      
      String nodeStr;
      String[] nodeInfo;
      Node node;
      
      long groupNum = key.get();
      
      for(Text val : values) {
        nodeStr = val.toString();
        
        nodeInfo = nodeStr.split("\\s+");
        
        node = new Node(groupNum,
                             Long.parseLong(nodeInfo[0]),
                             Long.parseLong(nodeInfo[1]));
        reduceLog.info("Node number: " + node.nodeNum);
        reduceLog.info("Node label: " + node.nodeLabel);
        Long nn = new Long(node.nodeNum);
        Long nl = new Long(node.nodeLabel);
        
        if( positionsMap.get(nn) == null ){
          positionsMap.put(nn, new ArrayList<Node>());
        }
        positionsMap.get(nn).add(node);
        
        if( labelsMap.get(nl) == null ){
          labelsMap.put(nl, new ArrayList<Node>());
        }
        labelsMap.get(nl).add(node);
      }
      
      List<Long> positions=new ArrayList<Long>(positionsMap.keySet());
      Collections.sort(positions);
      List<Long> labels=new ArrayList<Long>(labelsMap.keySet());
      Collections.sort(labels);
      
      // Loop through labels, performing DFS where necessary
      for(long i = 0; (int)i < labels.size(); i++){
          reduceLog.info("Processing label: " + labels.get((int)i));
          Long label = labels.get((int)i);
          if( !(labelsMap.get(label).get(0).visited) ){
              reduceLog.info("Need to do a DFS");
              dfs(label.longValue(), label.longValue(), positionsMap, labelsMap);
          }
      }
      
      // Loop through positions, outputting what we've found
      for(long k = 0; (int)k < positions.size(); k++){
          Long position = positions.get((int)k);
          ArrayList<Node> nodes = positionsMap.get(position);
          for(int l = 0; l < nodes.size(); l++){
              node = nodes.get(l);
              context.write(new LongWritable(node.nodeNum), 
                new LongWritable(node.nodeLabel)); 
          }
      }
      reduceLog.info("--------------------------------------"); 
    }
  }
  	
  public static void main(String[] args) throws Exception {

    Configuration conf = new Configuration();
        
    Job job = new Job(conf, "thirdpass");

    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(Text.class);

    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(LongWritable.class);
        
    job.setMapperClass(Map.class);
    job.setReducerClass(Reduce.class);
        
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
           
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileInputFormat.addInputPath(job, new Path(args[1]));
    FileOutputFormat.setOutputPath(job, new Path(args[2]));
        
    job.waitForCompletion(true); 
    /*
    JobConf conf = new JobConf(FirstPass.class);
    conf.setJobName("firstpass");
    
    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(LongWritable.class);
    
    conf.setMapperClass(Map.class);
    //conf.setCombinerClass(Reduce.class);
    conf.setReducerClass(Reduce.class);
    
    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);
    
    conf.setInputPath(new Path(args[0]));
    conf.setOutputPath(new Path(args[1]));
    //FileInputFormat.setInputPaths(conf, new Path(args[0]));
    //FileInputFormat.addInputPath(job, input);
    //FileOutputFormat.setOutputPath(conf, new Path(args[1]));
    
    JobClient.runJob(conf);
    */
  }
}
