import java.util.List;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class DetectAnomalies {
	
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
	
	private Dataset<Row> features;
	private Dataset<Row> dataset;

	private String featuresOrModelFile;
	private String datasetFile;
	private String modelNametoSave;
	private String[] featureColumns;
	
	private GBTClassifier classifier;
	private GBTClassificationModel model;
	
	public DetectAnomalies() {
		File file;
		Scanner in = new Scanner(System.in);
		System.out.println("Enter the name of the file you want to check for possible errors: ");
		datasetFile = in.nextLine();
		file = new File(datasetFile);
		while(!file.exists()){
			System.out.println("Enter the name of the file you want to check for possible errors: ");
			datasetFile = in.nextLine();
			file = new File(datasetFile);
		}
		dataset = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(datasetFile);
	
		dataset.cache();
	
		System.out.println("Enter the model you want to use for prediction OR enter the features file if you want to create a new model: ");
		featuresOrModelFile = in.nextLine();
		file = new File(featuresOrModelFile);
		while(!file.exists()){
			System.out.println("Enter the model you want to use for prediction OR enter the features file if you want to create a new model: ");
			featuresOrModelFile = in.nextLine();
			file = new File(featuresOrModelFile);
		}
		
		try{
			model = GBTClassificationModel.load(featuresOrModelFile);
			featureColumns = VectorAssembler.load(featuresOrModelFile + "assembler").getInputCols();
			
			this.prepareDataset();
		}catch(Exception e){
			e.printStackTrace();
			features = sparkSession.read()
					  .option("header", "true")
					  .option("inferSchema", "true")
					  .option("delimiter", "\t")
					  .csv(featuresOrModelFile);
			
			features.cache();
			featureColumns = features.columns();
			
			System.out.println("No model detected... Creating new model based on features entered.");
			System.out.println("Enter the name of the model to save: ");
			modelNametoSave = in.nextLine();
			dataset = dataset.withColumn("label", functions.lit(0));
			this.labelAnomalies();
			this.prepareDataset();
			this.createModel();
		}finally{
			in.close();
		}
	}
	
	private void labelAnomalies(){
		
		for(String column : featureColumns){
			Column col = new Column(column);
			Column label = new Column("label");
			List<Row> colFeatures = features.select(col).filter(col.isNotNull()).collectAsList();
			for(Row row : colFeatures){
				String feature = row.getAs(0);
				Column condition = this.getCondition(feature, col);
				dataset = dataset.withColumn("label", functions.when(condition.or(label.equalTo(1)), 1).otherwise(0));
			}
		}
	}
	
	private Column getCondition(String feature, Column col){
		Column condition = col.isNull();
		char filter = feature.charAt(0);
		Object value = feature.substring(1);
		
		if(feature.length() > 1){
			switch(filter){
			case '=':
				condition = col.equalTo(value);
				break;
			case '!':
				condition = col.notEqual(value);
				break;
			case '>':
				condition = col.gt(value);
				break;
			case '<':
				condition = col.lt(value);
				break;
			case '%':
				condition = col.like("%"+value+"%");
				break;
			default:
				condition = col.contains(value);
				break;
			}
		}
		
		return condition;
	}
	//prepares the dataset for features on the specified columns 
	private void prepareDataset(){
		VectorAssembler featureAssembler;
		List<String> featureCols = new ArrayList<>();
		List<String> columnsToCheck = new ArrayList<>();
		
		StringIndexerModel indexer;
		
		for(String column : featureColumns){
			if(dataset.head().getAs(column).getClass().getSimpleName().equals("String")){
				indexer = new StringIndexer().setInputCol(column).setOutputCol("INDEX"+column).fit(dataset);
				dataset = indexer.transform(dataset);
				featureCols.add("INDEX"+column);
			}
			else if(dataset.head().getAs(column).getClass().getSimpleName().equals("Timestamp")){
				dataset = dataset.withColumn("INDEX"+column, new Column(column).cast("Integer"));
				featureCols.add("INDEX"+column);
			}
			else
				featureCols.add(column);
			
			columnsToCheck.add(column);
		}
		
		String[] features = featureCols.toArray(new String[0]);
		featureAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		dataset = featureAssembler.transform(dataset);
		
		try {
			new VectorAssembler()
			.setInputCols(columnsToCheck.toArray(new String[0]))
			.setOutputCol("features")
			.save(modelNametoSave+"assembler");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//checks for anomalous data features the model on the whole dataset
	private void createModel(){
		classifier = new GBTClassifier();
		model = classifier.fit(dataset);
		try {
			model.save(modelNametoSave);
			System.out.println("Model saved!");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//displaus the possible anomalies to the user  
	public void printAnomalies(){
		Column anomaly = new Column("prediction");
		Dataset<Row> anomalies = model.transform(dataset).filter(anomaly.equalTo(1));
		System.out.println("Possible anomalous data: ");
		anomalies.select("prediction", featureColumns).drop("prediction").show(50);
		//System.out.println(model.toDebugString());
	}
}
