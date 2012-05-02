package org.myorg;
	
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
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

  public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, LongWritable> {
      private final static IntWritable one = new IntWritable(1);
      
      private Text word = new Text();

      public void map(LongWritable key, Text value, OutputCollector<LongWritable, LongWritable> output, Reporter reporter) throws IOException {
        
        double val = Double.parseDouble(value.toString());
        
        long lineNum = key.get() / 12 + 1;
        
        // Make sure m is actually going to be a point in our graph
        if(lineNum > m*m || val < wMin || val >= wLimit){
          return;
        }
        
        reduceLog.info("Line Number: " + Long.toString(lineNum));
        
        long x, y;
        
        long sqrt = (long) Math.ceil(Math.sqrt((double) lineNum));
        
        if( lineNum % sqrt == 0 ){
          x = sqrt - 1;
          y = sqrt - 1;
        } else {
          if((sqrt % 2) == (lineNum % 2)){
            y = sqrt - 1;
            x = sqrt - (1 + (long)(Math.ceil( ((double) sqrt * sqrt - lineNum) / 2)));
          } else {
            x = sqrt - 1;
            y = sqrt - (1 + (long)(Math.ceil( ((double) sqrt * sqrt - lineNum) / 2)));
          }
        }
        reduceLog.info("X: " + Long.toString(x));
        reduceLog.info("Y: " + Long.toString(y));
        
        reduceLog.info("Group Number: " + Long.toString(x/g));
        reduceLog.info("Number: " + Long.toString(x*m + y));
        
        output.collect(new LongWritable(x/g), new LongWritable(x*m + y));
        // Make sure we map boundary columns twice
        if(x != 0 && (x % g) == 0){
          output.collect(new LongWritable(x/g - 1), new LongWritable(x*m + y));
        }
        
        return;
      }
  }
	
  public static class Reduce extends MapReduceBase implements Reducer<LongWritable,LongWritable,LongWritable,Text> {
    
    private static void dfs(long i, long l, long[] elements, long[] label){
      label[(int)i] = l;
      
      if(i % m != (m - 1) && i != elements.length){
        if(elements[(int)(i + 1)] != 0 && label[(int)(i + 1)] == 0){
          dfs(i+1, l, elements, label);
        }
      }
      
      if(i % m != 0 && i != 0){
        if(elements[(int)(i - 1)] != 0  && label[(int)(i - 1)] == 0){
          dfs(i-1, l, elements, label);
        }
      }
      
      if(i >= m){
        if(elements[(int)(i - m)] != 0  && label[(int)(i - m)] == 0){
          dfs(i-m, l, elements, label);
        }
      }
      
      if(i < elements.length - m){
        if(elements[(int)(i + m)] != 0  && label[(int)(i + m)] == 0){
          dfs(i+m, l, elements, label);
        }
      }
      
      return;
    }
    
    public void reduce(LongWritable key, Iterator<LongWritable> values, OutputCollector<LongWritable, Text> output, Reporter reporter) throws IOException {
      
      long groupNum = key.get();
      // Size of first group (g = 0) is g*m, otherwise (1 + g) * m
      long maxElements = (g + 1)*m;
      long offset = groupNum * m * g - m;
      if(groupNum == 0){ maxElements -= m; offset += m;}
      
      long[] elements = new long[(int)maxElements];
      long[] label = new long[(int)maxElements];
      
      reduceLog.info("Group Number: " + Long.toString(groupNum));
      reduceLog.info("Max Elements: " + Long.toString(maxElements));
      reduceLog.info("Offset: " + Long.toString(offset));
      reduceLog.info("Length of elements array: " + Integer.toString(elements.length));
      
      while (values.hasNext()) {
        int index = (int)(values.next().get() - offset);
        reduceLog.info("Index: " + Integer.toString(index));
        elements[index] = 1;
      }
      
      for(long i = 0; i < maxElements; i++){
        if(elements[(int)i] == 1 && label[(int)i] == 0){
          // Perform a DFS 
          dfs(i, i, elements, label);
        }
        if(elements[(int)i] == 1){
          // Emit if on a boundary column
          if(i < m && groupNum != 0){
            output.collect(new LongWritable(groupNum),
              new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
          } else if(groupNum == 0 && i >= g*(m-1)){
            output.collect(new LongWritable(groupNum),
              new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
          } else if(groupNum != m/g && i >= g*m){
            output.collect(new LongWritable(groupNum),
              new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
          }
        }
      }
      // OUTLINE:
      // Read all values into array - NEED TO USE OFFSET
      //   - Size will be equal to (1 + g) * m, except for the first set, which
      //     is just g * m
      // Initialize another array of equal size, again corresponding to the integers
      //   - This will mark the connected component of the above elements
      // Starting from lowest, perform a depth first search
      //   - Be careful on values where val % g = 0 || val % g = g - 1
      //   - Keep track of lowest node on this search, (will be the first one),
      //     and mark each other node with this in the second array (without the offset)
      // Then go to the lowest node that is not marked in the second array, repeat
      // process until you pass all nodes
      //
      // Be care with values where val = m (these won't be in boundary)
      
    }
  }
  	
  public static void main(String[] args) throws Exception {
    JobConf conf = new JobConf(FirstPass.class);
    conf.setJobName("firstpass");
    
    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(LongWritable.class);
    
    conf.setMapperClass(Map.class);
    conf.setCombinerClass(Reduce.class);
    conf.setReducerClass(Reduce.class);
    
    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);
    
    FileInputFormat.setInputPaths(conf, new Path(args[0]));
    FileOutputFormat.setOutputPath(conf, new Path(args[1]));
    
    JobClient.runJob(conf);
  }
}
