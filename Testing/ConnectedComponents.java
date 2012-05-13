package cs5300;
	
import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.*;

//import org.apache.commons.logging.*;

public class ConnectedComponents {

  // Logging
  //static Log mapLog = LogFactory.getLog(FirstPassMap.class);
  //static Log reduceLog = LogFactory.getLog(FirstPassReduce.class);  

  // g^2 = m is optimal
  static int computeG(int val) {

      int estG = (int)(Math.pow((double)val, 1.5));
      int max = (int)(Math.ceil(Math.sqrt((double)val)));

      List<Integer> factors  = new ArrayList<Integer>();
      for(int i=1; i <= max; i++)
      {
          if(val % i == 0)
          {
              factors.add(i);
          }
      }

      return factors.get(factors.size() - 1);
  }
  
  public static class Node{
    public int groupNum;
    public int nodeNum;
    public int nodeLabel;
    public int numEdges;
    public boolean visited;
    
    public Node(int gn, int nn, int nl, int ne){
      groupNum = gn;
      nodeNum = nn;
      nodeLabel = nl;
      numEdges = ne;
      visited = false;
    }
  }
  
  // compute filter parameters for netid jsh263
  private static final float fromNetID = 0.362f;
  private static final float desiredDensity = 0.59f;
  private static final float wMin = 0.4f * fromNetID;
  private static final float wLimit = wMin + desiredDensity;

  public static class FirstPassMap extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
      
      private Text word = new Text();
      
      public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        
        int m = context.getConfiguration().getInt("m", 0);
        int g = context.getConfiguration().getInt("g", 0);
        float wMin = context.getConfiguration().getFloat("wMin", 0);
        float wLimit = context.getConfiguration().getFloat("wLimit", 0);
        
        float val = Float.parseFloat(value.toString());
        
        int lineNum = (int)(key.get() / 12 + 1);
        /*
        System.out.println("-----------------------------------------");
        System.out.println("FIRST MAP CALLED");
        System.out.println("Text seen is: " + value.toString());
        System.out.println("m: " + m);
        System.out.println("g: " + g);
        System.out.println("Line number: " + lineNum);
        System.out.println("-----------------------------------------");
        */
        // Make sure m is actually going to be a point in our graph
        if(lineNum > m*m || val < wMin || val >= wLimit){
          return;
        }
        
        int x, y;
        
        int sqrt = (int) Math.ceil(Math.sqrt((double) lineNum));
        
        // Calculate x/y coordinates of this point
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

        context.write(new IntWritable(x/g), new IntWritable(x*m + y));
        // If this is a boundary point
        if(x != (m - 1) && (x % g) == (g - 1)){
          context.write(new IntWritable(x/g + 1), new IntWritable(x*m + y));
        }
        /*
        //mapLog.info("-----------------------------------------");
        //mapLog.info("MAP IS WORKING");
        //mapLog.info("-----------------------------------------");
        System.out.println("-----------------------------------------");
        System.out.println("FIRST MAP COMPLETED SINGLE MAPPING");
        System.out.println("-----------------------------------------");
        */
        return;
      }
  }
	
    public static class FirstPassReduce extends Reducer<IntWritable,IntWritable,IntWritable,Text> {
    
    /*
    // Depth first labeling for the nodes
    private static void dfs(int m, int i, int l, int[] elements, int[] label){
      label[i] = l;
      if(i % m != (m - 1) && i != elements.length){
        if(elements[(i + 1)] != 0 && label[(i + 1)] == -1){
          dfs(m, i+1, l, elements, label);
        }
      }
      
      if(i % m != 0 && i != 0){
        if(elements[(i - 1)] != 0  && label[(i - 1)] == -1){
          dfs(m, i-1, l, elements, label);
        }
      }
      
      if(i >= m){
        if(elements[(i - m)] != 0  && label[(i - m)] == -1){
          dfs(m, i-m, l, elements, label);
        }
      }
      
      if(i < elements.length - m){
        if(elements[(i + m)] != 0  && label[(i + m)] == -1){
          dfs(m, i+m, l, elements, label);
        }
      }
      
      return;
    } */
    
    // Depth first labeling for the nodes
    private static void dfs(int m, int i, int l, int[] elements, int[] label){
      //label[i] = l;
      
      Stack stack = new Stack();
      stack.push(new Integer(i));
      
      while(!stack.empty()){
          i = ((Integer)stack.pop()).intValue();
          label[i] = l;
          
          if(i % m != (m - 1) && i != elements.length){
            if(elements[(i + 1)] != 0 && label[(i + 1)] == -1){
              stack.push(new Integer(i+1));
            }
          }
          if(i % m != 0 && i != 0){
            if(elements[(i - 1)] != 0  && label[(i - 1)] == -1){
              stack.push(new Integer(i-1));
            }
          }
          if(i >= m){
            if(elements[(i - m)] != 0  && label[(i - m)] == -1){
              stack.push(new Integer(i-m));
            }
          }
          if(i < elements.length - m){
            if(elements[(i + m)] != 0  && label[(i + m)] == -1){
              stack.push(new Integer(i+m));
            }
          }
      }
      
      return;
    }
    
    // count edges for a node
    private static void countEdges(int g,  int m, int i, int[] elements, int[] edgeCount){
    
      if(!(i < m && elements.length != g*m) ){ // So that we do not double count boundary edges
                                               // Don't count up and down edges if in
                                               //    the first column for all groups
                                               //    except group 0
        if(i % m != (m - 1) && i != elements.length){
          if(elements[(i + 1)] != 0){
            edgeCount[i] += 1;
          }
        }
        
        if(i % m != 0 && i != 0){
          if(elements[(i - 1)] != 0){
            edgeCount[i] += 1;
          }
        }
      }
      if(i >= m){
        if(elements[(i - m)] != 0){
          edgeCount[i] += 1;
        }
      }
      
      if(i < elements.length - m){
        if(elements[(i + m)] != 0){
          edgeCount[i] += 1;
        }
      }
      
      return;
    }
    
    public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
      
      int m = context.getConfiguration().getInt("m", 0);
      int g = context.getConfiguration().getInt("g", 0);
      
      int groupNum = key.get();
      // Size of first group (g = 0) is g*m, otherwise (1 + g) * m
      int maxElements = (g + 1)*m;
      int offset = groupNum * m * g - m;
      if(groupNum == 0){ maxElements -= m; offset += m;}
      
      int[] elements = new int[maxElements];
      int[] label = new int[maxElements];
      int[] edgeCount = new int[maxElements];
      Arrays.fill(label, (int)(-1));
      
      for(IntWritable val : values) {
        int index = val.get() - offset;
        elements[index] = 1;
      }
      int elementCount = 0;
      for(int i = 0; i < maxElements; i++){
        if(elements[i] == 1 && label[i] == -1){
          // Perform a DFS 
          dfs(m, i, i, elements, label);
        }
        if(elements[i] == 1){
          elementCount++;
          // Count the number of edges for a node
          countEdges(g, m, i, elements, edgeCount);
          // Emit if node exists 
          context.write(new IntWritable(groupNum),
              new Text(Integer.toString(i+offset) + " " + Integer.toString(label[i] + offset)
                        + " " + Integer.toString(edgeCount[i]))); 
        }
      }
      System.out.println(elementCount);
    }
  }
  	
  public static class SecondPassMap extends Mapper<LongWritable, Text, IntWritable, Text> { 
      
      public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        
        int m = context.getConfiguration().getInt("m", 0);
        int g = context.getConfiguration().getInt("g", 0);
        
        IntWritable one = new IntWritable(1);
        
        String nodeStr = value.toString();
        
        String[] node = nodeStr.split("\\s+");
        
        int groupNum = Integer.parseInt(node[0]);
        int nodeNum = Integer.parseInt(node[1]);
        int nodeLabel = Integer.parseInt(node[2]);
        int numEdges = Integer.parseInt(node[3]);
        
        int gm = g*m;
        int modulus = nodeNum % gm;
        
        // First find out if the node is a boundary node
        if( (nodeNum < m*(m - 1)) && (modulus >= (g-1)*m) ){
          context.write(one, new Text(nodeStr));
        }
        
        return;
      }
  }
  
  public static class SecondPassReduce extends Reducer<IntWritable,Text,IntWritable,Text> {
    
    /*
    // Depth first labeling of nodes
    private static void dfs(int currentLabel, int newLabel,
      HashMap<Integer, ArrayList<Node>> positionsMap, HashMap<Integer, ArrayList<Node>> labelsMap){
        Integer key = new Integer(currentLabel);
        ArrayList<Node> nodes = labelsMap.get(key);
        
        // Iterate through each position in the labels list, running dfs
        // on each of those labels
        for(int i = 0; i < nodes.size(); i++){
            nodes.get(i).nodeLabel = newLabel;
            nodes.get(i).visited = true;
        }
        
         for(int i = 0; i < nodes.size(); i++){
            Integer nodeNum = new Integer(nodes.get(i).nodeNum);
            ArrayList<Node> nodesWithNodeNum = positionsMap.get(nodeNum);
            for(int j = 0; j < nodesWithNodeNum.size(); j++){
                if(!(nodesWithNodeNum.get(j).visited)){
                    dfs(nodesWithNodeNum.get(j).nodeLabel, newLabel, positionsMap, labelsMap);
                }
            }
        }
    }
    */
    
    // Depth first labeling of nodes
    private static void dfs(int currentLabel, int newLabel,
      HashMap<Integer, ArrayList<Node>> positionsMap, HashMap<Integer, ArrayList<Node>> labelsMap){
        
        Integer key = new Integer(currentLabel);
        
        Stack stack = new Stack();
        stack.push(key);
        
        while(!stack.empty()){
        
            key = ((Integer)stack.pop());
            ArrayList<Node> nodes = labelsMap.get(key);
            
            
            for(int i = 0; i < nodes.size(); i++){
                nodes.get(i).nodeLabel = newLabel;
                nodes.get(i).visited = true;
            }
            
             for(int i = 0; i < nodes.size(); i++){
                Integer nodeNum = new Integer(nodes.get(i).nodeNum);
                ArrayList<Node> nodesWithNodeNum = positionsMap.get(nodeNum);
                for(int j = 0; j < nodesWithNodeNum.size(); j++){
                    if(!(nodesWithNodeNum.get(j).visited)){
                        stack.push(new Integer(nodesWithNodeNum.get(j).nodeLabel));
                    }
                }
            }
        }
        
        return;
    }
    
    public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      
      int m = context.getConfiguration().getInt("m", 0);
      int g = context.getConfiguration().getInt("g", 0);
      
      // Hashmaps with an appropriate initial capacity
      HashMap<Integer, ArrayList<Node>> positionsMap = new HashMap<Integer, ArrayList<Node>>(m*m/g);
      HashMap<Integer, ArrayList<Node>> labelsMap = new HashMap<Integer, ArrayList<Node>>(m*m/g);
      
      String nodeStr;
      String[] nodeInfo;
      Node node;
      
      for(Text val : values) {
        nodeStr = val.toString();
        
        nodeInfo = nodeStr.split("\\s+");
        
        node = new Node(Integer.parseInt(nodeInfo[0]),
                             Integer.parseInt(nodeInfo[1]),
                             Integer.parseInt(nodeInfo[2]),
                             Integer.parseInt(nodeInfo[3]));
        Integer nn = new Integer(node.nodeNum);
        Integer nl = new Integer(node.nodeLabel);
        
        if( positionsMap.get(nn) == null ){
          positionsMap.put(nn, new ArrayList<Node>());
        }
        positionsMap.get(nn).add(node);
        
        if( labelsMap.get(nl) == null ){
          labelsMap.put(nl, new ArrayList<Node>());
        }
        labelsMap.get(nl).add(node);
      }
      
      List<Integer> positions=new ArrayList<Integer>(positionsMap.keySet());
      Collections.sort(positions);
      List<Integer> labels=new ArrayList<Integer>(labelsMap.keySet());
      Collections.sort(labels);
      
      // Loop through labels, performing DFS where necessary
      for(int i = 0; i < labels.size(); i++){
          Integer label = labels.get(i);
          if( !(labelsMap.get(label).get(0).visited) ){
              dfs(label.intValue(), label.intValue(), positionsMap, labelsMap);
          }
      }
      
      // Loop through positions, outputting what we've found
      for(int k = 0; k < positions.size(); k++){
          Integer position = positions.get(k);
          ArrayList<Node> nodes = positionsMap.get(position);
          // Get edge counts for each node
          int totalEdges = 0;
          for(int l = 0; l < nodes.size(); l++){
              totalEdges += nodes.get(l).numEdges;
          }
          
          for(int l = 0; l < nodes.size(); l++){
              node = nodes.get(l);
              context.write(new IntWritable(node.groupNum), 
                new Text(Integer.toString(node.nodeNum) + " " + Integer.toString(node.nodeLabel) + 
                         " " + Integer.toString(totalEdges))); 
          }
      }
    }
  }
  
  public static class ThirdPassMap extends Mapper<LongWritable, Text, IntWritable, Text> { 
      
      public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        
        int m = context.getConfiguration().getInt("m", 0);
        int g = context.getConfiguration().getInt("g", 0);
        
        IntWritable one = new IntWritable(1);
        
        String nodeStr = value.toString();
        
        String[] node = nodeStr.split("\\s+");
        
        int groupNum = Integer.parseInt(node[0]);
        int nodeNum = Integer.parseInt(node[1]);
        int nodeLabel = Integer.parseInt(node[2]);
        int numEdges = Integer.parseInt(node[3]);
        
        int gm = g*m;
        int modulus = nodeNum % gm;
        
        context.write(new IntWritable(groupNum), new Text(node[1] + " " + node[2] + " " + node[3]));
        
        return;
      }
  }
  
  public static class ThirdPassReduce extends Reducer<IntWritable,Text,IntWritable,Text> {
    
    /*
    private static void dfs(int currentLabel, int newLabel,
      HashMap<Integer, ArrayList<Node>> positionsMap, HashMap<Integer, ArrayList<Node>> labelsMap){
        Integer key = new Integer(currentLabel);
        ArrayList<Node> nodes = labelsMap.get(key);
        
        // Iterate through each position in the labels list, running dfs
        // on each of those labels
        for(int i = 0; i < nodes.size(); i++){
            nodes.get(i).nodeLabel = newLabel;
            nodes.get(i).visited = true;
        }
        
         for(int i = 0; i < nodes.size(); i++){
            Integer nodeNum = new Integer(nodes.get(i).nodeNum);
            ArrayList<Node> nodesWithNodeNum = positionsMap.get(nodeNum);
            for(int j = 0; j < nodesWithNodeNum.size(); j++){
                if(!(nodesWithNodeNum.get(j).visited)){
                    dfs(nodesWithNodeNum.get(j).nodeLabel, newLabel, positionsMap, labelsMap);
                }
            }
        }
    }
    */
    
    private static void dfs(int currentLabel, int newLabel,
      HashMap<Integer, ArrayList<Node>> positionsMap, HashMap<Integer, ArrayList<Node>> labelsMap){
        Integer key = new Integer(currentLabel);
        
        Stack stack = new Stack();
        stack.push(key);
        
        while(!stack.empty()){
        
            key = ((Integer)stack.pop());
            ArrayList<Node> nodes = labelsMap.get(key);
            
            // Iterate through each position in the labels list, running dfs
            // on each of those labels
            for(int i = 0; i < nodes.size(); i++){
                nodes.get(i).nodeLabel = newLabel;
                nodes.get(i).visited = true;
            }
            
             for(int i = 0; i < nodes.size(); i++){
                Integer nodeNum = new Integer(nodes.get(i).nodeNum);
                ArrayList<Node> nodesWithNodeNum = positionsMap.get(nodeNum);
                for(int j = 0; j < nodesWithNodeNum.size(); j++){
                    if(!(nodesWithNodeNum.get(j).visited)){
                        stack.push(new Integer(nodesWithNodeNum.get(j).nodeLabel));
                    }
                }
            }
        }
    }
    
    public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      
      int m = context.getConfiguration().getInt("m", 0);
      int g = context.getConfiguration().getInt("g", 0);
      
      // Hashmaps with an appropriate initial capacity
      HashMap<Integer, ArrayList<Node>> positionsMap = new HashMap<Integer, ArrayList<Node>>((m*m/g));
      HashMap<Integer, ArrayList<Node>> labelsMap = new HashMap<Integer, ArrayList<Node>>((m*m/g));
      
      String nodeStr;
      String[] nodeInfo;
      Node node;
      
      int groupNum = key.get();
      
      for(Text val : values) {
        nodeStr = val.toString();
        
        nodeInfo = nodeStr.split("\\s+");
        
        node = new Node(groupNum,
                             Integer.parseInt(nodeInfo[0]),
                             Integer.parseInt(nodeInfo[1]),
                             Integer.parseInt(nodeInfo[2]));
        int nn = new Integer(node.nodeNum);
        int nl = new Integer(node.nodeLabel);
        
        if( positionsMap.get(nn) == null ){
          positionsMap.put(nn, new ArrayList<Node>());
        }
        positionsMap.get(nn).add(node);
        
        if( labelsMap.get(nl) == null ){
          labelsMap.put(nl, new ArrayList<Node>());
        }
        labelsMap.get(nl).add(node);
      }
      
      List<Integer> positions=new ArrayList<Integer>(positionsMap.keySet());
      Collections.sort(positions);
      List<Integer> labels=new ArrayList<Integer>(labelsMap.keySet());
      Collections.sort(labels);
      
      // Loop through labels, performing DFS where necessary
      for(int i = 0; i < labels.size(); i++){
          Integer label = labels.get(i);
          if( !(labelsMap.get(label).get(0).visited) ){
              dfs(label.intValue(), label.intValue(), positionsMap, labelsMap);
          }
      }
      
      // Loop through positions, outputting what we've found
      for(int k = 0; k < positions.size(); k++){
          Integer position = positions.get(k);
          ArrayList<Node> nodes = positionsMap.get(position);
          int edgeCount = 0;
          for(int l = 0; l < nodes.size(); l++){
              node = nodes.get(l);
              if(node.numEdges > edgeCount){
                  edgeCount = node.numEdges;
              }
          }
          node = nodes.get(0);
          /*
          int gm = g*m;
          int modulus = nodeNum % gm;
          boolean isLowerBoundary = nodeNum < m*(m - 1))
                                    && (modulus >= (g-1)*m)
                                    && nodeNum
          */
          // First find out if the node is a boundary node
          int offset = groupNum * m * g - m;
          int offsetNodeNum = node.nodeNum - offset;
          if(!(offsetNodeNum < m && groupNum != 0)){

            context.write(new IntWritable(node.nodeNum), 
                new Text(Integer.toString(node.nodeLabel) + " " +
                         Integer.toString(edgeCount)));
          }
          
      }
    }
  }
  
  public static class StatisticsMap extends Mapper<LongWritable, Text, IntWritable, Text> {
	  public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {	
      IntWritable one = new IntWritable(1);
		  String line = value.toString();
		  context.write(one, new Text(line));
		}
	}

  public static class StatisticsReduce extends Reducer<IntWritable, Text, IntWritable, Text> {
	public void reduce(IntWritable key, Iterable<Text> iterableValues, Context context) throws IOException, InterruptedException {
    
    int m = context.getConfiguration().getInt("m", 0);
    int g = context.getConfiguration().getInt("g", 0);
    
    IntWritable one = new IntWritable(1);

		Iterator<Text> values = iterableValues.iterator();

		int totalNodes = 0;
		int totalEdges = 0;
		HashMap<Integer, Integer> components = new HashMap<Integer, Integer>();

		while (values.hasNext()) {
		  String val = values.next().toString();
			String[] line = val.split("\\s+");
			Integer label = Integer.parseInt(line[1]);
			Integer edges = Integer.parseInt(line[2]);
			int node = components.containsKey(label) ? components.get(label) : 0;
			totalEdges += edges;
			totalNodes++;
			components.put(label, node + 1);
		}
		totalEdges /= 2; // Each edge is counted twice so divide by 2 to get real total
		int totalComponents = components.size();

		float sum = 0;
		for (Integer ccSize : components.values()) {
			sum += ccSize * ccSize;
		}

		float averageBurnCount = sum / (m*m);
		float averageComponentSize = sum / totalNodes;

		String s = "Number of vertices: %s \n" + 
					"Number of edges: %s \n" + 
					"Number of distinct connected components: %s \n" + 
					"Average size of connected components: %s \n" + 
					"Average burn count: %s \n";
		System.out.println(s);
		String stats = String.format(s, Integer.toString(totalNodes), Integer.toString(totalEdges), 
			Integer.toString(totalComponents), Float.toString(averageComponentSize), Float.toString(averageBurnCount));

		context.write(one, new Text(stats));
	}
  }
  
  public static void main(String[] args) throws Exception {

    // Format for folders on S3:
    // s3n://(file|folder)
    // Order of args: m inputPath outputPath
    
    /*
    cs5300.ConnectedComponents 40 s3n://jasdeep/input/data11.txt s3n://jasdeep/output
    cs5300.ConnectedComponents 20000 s3n://edu-cornell-cs-cs5300s12-assign5-data/production.txt s3n://jasdeep/output
    */

    int m = Integer.parseInt(args[0]);
    int g = computeG(m);

    String input = args[1];
    String output = args[2];

    String  first = "FirstPassOutput";
    String  second = "SecondPassOutput";
    String  third = "ThirdPassOutput";
    String  stats = "StatisticsOutput";
    //FileInputStream fstream = new FileInputStream(input);
    //DataInputStream in = new DataInputStream(fstream);
    //BufferedReader br = new BufferedReader(new InputStreamReader(in));
    //String strLine;
    //Read File Line By Line
    //while ((strLine = br.readLine()) != null)   {
      // Print the content on the console
      //System.out.println (strLine);
    //}
    System.out.println("-----------------------------------------");

    Configuration conf = new Configuration();
    conf.setFloat("wLimit", wLimit);
    conf.setFloat("wMin", wMin);
    conf.setInt("m", m);
    conf.setInt("g", g);
        
    Job job = new Job(conf, "firstpass");
    job.setJarByClass(ConnectedComponents.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(IntWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Text.class);        
    job.setMapperClass(FirstPassMap.class);
    job.setReducerClass(FirstPassReduce.class);        
    //job.setInputFormatClass(TextInputFormat.class);
    //job.setOutputFormatClass(TextOutputFormat.class);       
    FileInputFormat.addInputPath(job, new Path(input));
    FileOutputFormat.setOutputPath(job, new Path(output + "/" + first));
    job.waitForCompletion(true);
    
    job = new Job(conf, "secondpass");
    job.setJarByClass(ConnectedComponents.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Text.class);
    job.setMapperClass(SecondPassMap.class);
    job.setReducerClass(SecondPassReduce.class);
    //job.setInputFormatClass(TextInputFormat.class);
    //job.setOutputFormatClass(TextOutputFormat.class);
    FileInputFormat.addInputPath(job, new Path(output + "/" + first));
    FileOutputFormat.setOutputPath(job, new Path(output + "/" + second));
    job.waitForCompletion(true);
    
    job = new Job(conf, "thirdpass");
    job.setJarByClass(ConnectedComponents.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(IntWritable.class);
    job.setMapperClass(ThirdPassMap.class);
    job.setReducerClass(ThirdPassReduce.class);
    //job.setInputFormatClass(TextInputFormat.class);
    //job.setOutputFormatClass(TextOutputFormat.class);
    FileInputFormat.addInputPath(job, new Path(output + "/" + first));
    FileInputFormat.addInputPath(job, new Path(output + "/" + second));
    FileOutputFormat.setOutputPath(job, new Path(output + "/" + third));
    job.waitForCompletion(true); 
    
    job = new Job(conf, "statistics");
    job.setJarByClass(ConnectedComponents.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Text.class);
    job.setMapperClass(StatisticsMap.class);
    job.setReducerClass(StatisticsReduce.class);
    //job.setInputFormatClass(TextInputFormat.class);
    //job.setOutputFormatClass(TextOutputFormat.class);
    FileInputFormat.addInputPath(job, new Path(output + "/" + third));
    FileOutputFormat.setOutputPath(job, new Path(output + "/" + stats));
    job.waitForCompletion(true); 
  }
}
