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

public class FirstPass {

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
        
        // First find out if the node is a boundary node
        if( (nodeNum < m*(m - 1) && nodeNum % g*m >= g*(m-1)) ){
          context.write(one, nodeStr);
        }
        
        return;
      }
  }
  
  public static class Reduce extends Reducer<LongWritable,LongWritable,LongWritable,Text> {
    
    public class Node{
      public long groupNum;
      public long nodeNum;
      public long nodeLabel;
      
      public Node(gn, nn, nl){
        groupNum = gn;
        nodeNum = nn;
        nodeLabel = nl;
      }
    }
    
    public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      
      // Hashmaps with an appropriate initial capacity
      HashMap<Long, ArrayList<Node>> positionsMap = new HashMap<Long, ArrayList<Node>>(m*m/g);
      HashMap<Long, ArrayList<Node>> labelsMap = new HashMap<Long, ArrayList<Node>>(m*m/g);
      
      String nodeStr;
      String[] node;
      
      for(Text val : values) {
        nodeStr = val.toString();
        
        node = nodeStr.split("\\s+");
        
        Node node = new Node(Long.parseLong(node[0]),
                             Long.parseLong(node[1]),
                             Long.parseLong(node[2]));
        
        Long nn = new Long(node.nodeNum);
        Long nl = new Long(node.nodeLabel);
        
        if( positionsMap.get(nn) == null ){
          positionsMap.put(new ArrayList<Node>());
        }
        positionsMap.get(nn).add(node);
        
        if( labelsMap.get(nl) == null ){
          labelsMap.put(new ArrayList<Node>());
        }
        labelsMap.get(nl).add(node);
      }
      
      List positions=new ArrayList(positionsMap.keySet());
      Collections.sort(positions);
      List labels=new ArrayList(labelsMap.keySet());
      Collections.sort(labels);
      
      // Perform a DFS
      
    }
  }
  	
  public static void main(String[] args) throws Exception {

    Configuration conf = new Configuration();
        
    Job job = new Job(conf, "secondpass");

    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(LongWritable.class);

    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(Text.class);
        
    job.setMapperClass(Map.class);
    job.setReducerClass(Reduce.class);
        
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);
           
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
        
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