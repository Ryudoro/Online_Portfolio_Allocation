pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                script {
                    checkout scm
                }
            }
        }
        // stage('Install Python') {
        //     steps {
        //         script {
        //             sh 'apt-get update && apt-get install -y python3'
        //         }
        //     }
        // }
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Data Preparation and Model Training') {
            steps {
                sh 'python3 Model/input_creation.py'
                sh 'python3 Model/model_data_creation.py'
                sh 'python3 Model/model_creation.py'
            }
        }
        
        // stage('Predictions and Visualization') {
        //     steps {
        //         sh 'python make_predictions.py'
        //         sh 'python visualize_results.py'
        //     }
        // }
        
        stage('Tests') {
            steps {
                sh 'python3 Model/test_result.py'
            }
        }
        
        // stage('Deploy (Optional)') {
        //     when {
        //         branch 'master'
        //     }
        //     steps {
        //         sh 'python deploy.py'
        //     }
        // }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '**/*', allowEmptyArchive: true
        }
    }
}

