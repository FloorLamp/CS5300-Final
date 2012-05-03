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

//  public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, LongWritable> {
  public static class Map extends Mapper<LongWritable, Text, LongWritable, LongWritable> { 
      
      private static long countEmissions = 0;
      
      private Text word = new Text();

//      public void map(LongWritable key, Text value, OutputCollector<LongWritable, LongWritable> output, Reporter reporter) throws IOException {

      public void map(LongWritable key, Text value, Context context) throws IOException {
        
        double val = Double.parseDouble(value.toString());
        
        long lineNum = key.get() / 12 + 1;
        
        // Make sure m is actually going to be a point in our graph
        if(lineNum > m*m || val < wMin || val >= wLimit){
          return;
        }
        
        mapLog.info("Line Number: " + Long.toString(lineNum));
        
        long x, y;
        
        long sqrt = (long) Math.ceil(Math.sqrt((double) lineNum));
        mapLog.info("Ceiling of square root: " + Long.toString(sqrt));
        
        if(sqrt*sqrt == lineNum){
          x = sqrt - 1;
          y = sqrt - 1;
        } else {
          if((sqrt % 2) == (lineNum % 2)){
            y = sqrt - 1;
            x = sqrt - (1 + (sqrt * sqrt - lineNum) / 2);
          } else {
            x = sqrt - 1;
            y = sqrt - (1 + (sqrt * sqrt - lineNum + 1) / 2);
          }
        }
        mapLog.info("X: " + Long.toString(x));
        mapLog.info("Y: " + Long.toString(y));
        
        mapLog.info("Group Number: " + Long.toString(x/g));
        mapLog.info("Number: " + Long.toString(x*m + y));
        
        
        //output.collect(new LongWritable(x/g), new LongWritable(x*m + y));
        context.write(new LongWritable(x/g), new LongWritable(x*m + y));
        countEmissions++;
        // Make sure we map boundary columns twice
        if(x != (m - 1) && (x % g) == (g - 1)){
          //output.collect(new LongWritable(x/g + 1), new LongWritable(x*m + y));
          context.write(new LongWritable(x/g + 1), new LongWritable(x*m + y));
          countEmissions++;
        }
        
        mapLog.info("Total number of emissions: " + Long.toString(countEmissions));
        mapLog.info("-----------------------------------");
        
        return;
      }
  }
	
//  public static class Reduce extends MapReduceBase implements Reducer<LongWritable,LongWritable,LongWritable,Text> {

    public static class Reduce extends Reducer<LongWritable,LongWritable,LongWritable,Text> {
    
    private static void dfs(long i, long l, long[] elements, long[] label){
      label[(int)i] = l;
      //reduceLog.info("Label: " + Long.toString(label[(int)i]));
      if(i % m != (m - 1) && i != elements.length){
        if(elements[(int)(i + 1)] != 0 && label[(int)(i + 1)] == -1){
          dfs(i+1, l, elements, label);
        }
      }
      
      if(i % m != 0 && i != 0){
        if(elements[(int)(i - 1)] != 0  && label[(int)(i - 1)] == -1){
          dfs(i-1, l, elements, label);
        }
      }
      
      if(i >= m){
        if(elements[(int)(i - m)] != 0  && label[(int)(i - m)] == -1){
          dfs(i-m, l, elements, label);
        }
      }
      
      if(i < elements.length - m){
        if(elements[(int)(i + m)] != 0  && label[(int)(i + m)] == -1){
          dfs(i+m, l, elements, label);
        }
      }
      
      return;
    }
    
//    public void reduce(LongWritable key, Iterator<LongWritable> values, OutputCollector<LongWritable, Text> output, Reporter reporter) throws IOException {

    public void reduce(LongWritable key, Iterator<LongWritable> values, Context context) throws IOException {
      
      long groupNum = key.get();
      // Size of first group (g = 0) is g*m, otherwise (1 + g) * m
      long maxElements = (g + 1)*m;
      long offset = groupNum * m * g - m;
      if(groupNum == 0){ maxElements -= m; offset += m;}
      
      long[] elements = new long[(int)maxElements];
      long[] label = new long[(int)maxElements];
      Arrays.fill(label, (long)(-1));
      
      reduceLog.info("Group Number: " + Long.toString(groupNum));
      reduceLog.info("Max Elements: " + Long.toString(maxElements));
      reduceLog.info("Offset: " + Long.toString(offset));
      //reduceLog.info("Length of elements array: " + Integer.toString(elements.length));
      
      while (values.hasNext()) {
        int index = (int)(values.next().get() - offset);
        reduceLog.info("Index: " + Integer.toString(index));
        elements[index] = 1;
      }
      
      for(long i = 0; i < maxElements; i++){
        if(elements[(int)i] == 1 && label[(int)i] == -1){
          // Perform a DFS 
          dfs(i, i, elements, label);
        }
        if(elements[(int)i] == 1){
          // Emit if on a boundary column
          if(i < m && groupNum != 0){
            //output.collect(new LongWritable(groupNum),
            //  new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
            context.write(new LongWritable(groupNum),
              new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
          } else if(groupNum == 0 && i >= g*(m-1)){
            //output.collect(new LongWritable(groupNum),
            //  new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
            context.write(new LongWritable(groupNum),
              new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
          } else if(groupNum != m/g && i >= g*m){
            //output.collect(new LongWritable(groupNum),
            //  new Text(Long.toString(i+offset) + " " + Long.toString(label[(int)i] + offset)));
            context.write(new LongWritable(groupNum),
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
    //JobConf conf = JobBuilder.parseInputAndOutput(this,getConf(),args);
  
    Configuration conf = new Configuration();
        
    Job job = new Job(conf, "firstpass");

    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(Text.class);
        
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
