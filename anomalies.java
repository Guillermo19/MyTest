package org.apache.anomaly.detection;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class Anomalies {
	private LinearRegression linReg;
	private LinearRegressionModel model;
	
	private Dataset<Row> dataset;
	private Dataset<Row> vectorDataset;
	private Dataset<Row> residuals;
	
	private ArrayList<Row> anomalies;
	
	private final static double threshold = 3.0;
	
	private final static SparkConf sparkConf = 
			new SparkConf()
			.setAppName("Anomaly Detection")
			.setMaster("local");
	
	private final static JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
	
	private final static SparkSession sparkSession = 
			SparkSession
			  .builder()
			  .appName("Anomaly Detection")
			  .getOrCreate();
	
	public Anomalies(String filePath) {
	
		dataset = sparkSession.read().
				option("header", "true").
				option("delimiter", "\t").
				option("inferSchema", "true").
				csv(filePath);  //reads the data from a file whose path is passed to the constructor 
		
		dataset.cache(); 
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"Size"})
				.setOutputCol("features");
		
		vectorDataset = assembler.transform(dataset); //converts and combines the 'size' and 'records' column as the vector type 'features' column
		vectorDataset.cache();
		
		linReg = new LinearRegression().setLabelCol("Volume").setStandardization(true);
		
		model = linReg.fit(vectorDataset); //trains the k-means with the dataset to create a model
		model.summary().residuals().show();
		anomalies = new ArrayList<>();
		
	} //end constructor
	
	private boolean areThereAnomalies(){
		boolean areAnomalies = false;
		residuals = model.summary().residuals();
		Column residualCol = new Column("residuals");
		double residualStddev = sparkContext.parallelizeDoubles(residuals.toJavaRDD().map(f -> f.getDouble(0)).collect()).stdev();
		Dataset<Row> standardResiduals = residuals.withColumn("standardResiduals", residualCol.divide(residualStddev)).drop(residualCol);
		
		List<Row> stdResidList = standardResiduals.collectAsList();
		List<Row> list = vectorDataset.collectAsList();
		
		standardResiduals.show();
		for(int i = 0; i<stdResidList.size(); i++){
			double residual = stdResidList.get(i).getDouble(0);
			
			if(Math.abs(residual) > threshold){
				anomalies.add(list.get(i));
				areAnomalies = true;
				vectorDataset = vectorDataset.filter(new Column("File").notEqual(list.get(i).getAs("File")));
			}
		}
		
		return areAnomalies;
	}
	private void getAnomalies(){ //identifies potential outliers that might be mixed with normal data and saves them
		boolean anomalies = this.areThereAnomalies();
		while(anomalies){
			model = linReg.fit(vectorDataset);
			anomalies = this.areThereAnomalies();
		}
		
	}//end get anomalies
	
	public void printAnomalies() {
		this.getAnomalies();
		//vectorDataset.show(25,false);
		if(!anomalies.isEmpty()){
			System.out.println("Possible anomalous files: ");
			for(Row r : anomalies){
				System.out.println(r.getAs("File").toString());
			}
		}
		else{
			System.out.println("No probable anomalous files");
		}
		
	} //end print anomalies 
}
