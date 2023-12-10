import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


class KnnClassifier:

    def __init__(self, **kwargs):
        self.data_set = kwargs['data_set']
        self.original_data_frame = pd.read_csv(self.data_set)
        self.original_data_frame=self.original_data_frame.dropna()
        
        self.percents = kwargs.get('percents', [0.25, 0.25, 0.5])
        self.class_identifier = kwargs.get('class_identifier', 'class')

        self.split_data()

    def split_data(self):
        self.original_data_frame = shuffle(self.original_data_frame)
        df_size = len(self.original_data_frame)
        self.z1 = self.original_data_frame.iloc[:round(self.percents[0] * df_size)]
        self.z2 = self.original_data_frame.iloc[round(self.percents[0] * df_size):round((self.percents[0]+self.percents[1]) * df_size)]
        self.z3 = self.original_data_frame.iloc[round((self.percents[0]+self.percents[1]) * df_size):]        

    def find_best_k(self, z1, z2):
        k_values = list(range(1, len(z1)))
        scores = []

        X_train = z1.drop(self.class_identifier, axis=1)
        y_train = z1[self.class_identifier]

        X_test = z2.drop(self.class_identifier, axis=1)
        y_test = z2[self.class_identifier]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k) 
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            scores.append([np.mean(accuracy), k])

        scores.sort(reverse=True)

        # print(scores)

        return scores[0][1]

    def find_misclassified_indices(self, z1, z2, k):
        X_train = z1.drop(self.class_identifier, axis=1)
        y_train = z1[self.class_identifier]

        X_test = z2.drop(self.class_identifier, axis=1)
        y_test = z2[self.class_identifier]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)

        misclassified_indices = np.where(y_pred != y_test)[0]

        return misclassified_indices

    def swap_elements(self, z1, z2, indices_to_swap):
        for index_to_swap in indices_to_swap:
            random_index = np.random.choice(len(z1))
            z1.iloc[random_index], z2.iloc[index_to_swap] = z2.iloc[index_to_swap].copy(), z1.iloc[random_index].copy()

        return z1, z2

    def get_accuracy(self, z1, z2, k):
        X_train = z1.drop(self.class_identifier, axis=1)
        y_train = z1[self.class_identifier]

        X_test = z2.drop(self.class_identifier, axis=1)
        y_test = z2[self.class_identifier]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def method_1(self, z1, z2, iterations=100):
        k = self.k()
        scores = []
        z = []
        for i in range(iterations):
            wrong_elements = self.find_misclassified_indices(z1, z2, k)
            self.swap_elements(z1, z2, wrong_elements)
            scores.append([self.get_accuracy(z1, z2, k), i])
            z.append(z1.copy())
        scores.sort(reverse=True)
        z1 = z[scores[0][1]]
            
    def method_2(self, z1, z2, iterations=100):
        scores = []
        z = []
        for i in range(iterations):
            k = self.k()
            wrong_elements = self.find_misclassified_indices(z1, z2, k)
            self.swap_elements(z1, z2, wrong_elements)
            scores.append([self.get_accuracy(z1, z2, k), i])
            z.append(z1.copy())
        scores.sort(reverse=True)
        z1 = z[scores[0][1]]
            
    def method_3(self, z1, z2, iterations=100):
        scores = []
        z = []
        for i in range(iterations):
            k = self.k()
            wrong_elements = self.find_misclassified_indices(z1, z2, k)
            self.swap_elements(z1, z2, wrong_elements)
            scores.append([self.get_accuracy(z1, z2, k), i])
            z.append(z1.copy())
        scores.sort(reverse=True)
        z1 = z[scores[0][1]]
    
    def k(self):
        return self.find_best_k(self.z1, self.z2)

    def validate(self, **kwargs):
        method = kwargs.get('method', 'null')
        iterations = kwargs.get('iterations', 50)
        method_iterations = kwargs.get('method_iterations', 30)
        results = []

        for _ in range(iterations):
            self.split_data()
            z1 = self.z1.copy()
            z2 = self.z2.copy()
            z3 = self.z2.copy()

            if method == 'method_1':
                self.method_1(z1, z2, iterations=method_iterations)
            elif method == 'method_2':
                self.method_2(z1, z2, iterations=method_iterations)
            elif method != 'null':
                raise ValueError(f'Método desconhecido: {method}')

            accuracy = self.get_accuracy(z1, z3, self.find_best_k(z1, z3))
            results.append(accuracy)

        mean_accuracy = np.mean(results)
        std_dev_accuracy = np.std(results)
        confidence_level = 0.95
        z_value = norm.ppf((1 + confidence_level) / 2)

        margin_of_error = z_value * (std_dev_accuracy / np.sqrt(iterations))
        confidence_interval = (mean_accuracy - margin_of_error, mean_accuracy + margin_of_error)

        print(f'Média da acurácia: {mean_accuracy * 100:.2f}%')
        print(f'Desvio padrão da acurácia: {std_dev_accuracy * 100:.2f}%')
        print(f'Intervalo de Confiança: {confidence_interval}')

        return mean_accuracy, std_dev_accuracy, confidence_interval

    def cross_validate(self, **kwargs):
        iterations = kwargs.get('iterations', 50)
        method_iterations = kwargs.get('method_iterations', 30)

        results = []

        for _ in range(iterations):
            self.split_data()
            params = []
            z1_method_1 = self.z1.copy()
            z1_method_2 = self.z1.copy()

            self.method_1(z1_method_1, self.z2.copy(), iterations=method_iterations)
            self.method_2(z1_method_2, self.z2.copy(), iterations=method_iterations)

            for i in range(1, 16, 2):
                if i%3:
                    params.append((z1_method_1.copy(), i))
                else:
                    params.append((z1_method_2.copy(), i))

            y_pred = []

            X_test = self.z3.drop(self.class_identifier, axis=1)
            y_test = self.z3[self.class_identifier]

            for z1, k in params:

                X_train = z1.drop(self.class_identifier, axis=1)
                y_train = z1[self.class_identifier]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)

                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)

                y_pred.append(knn.predict(scaler.transform(X_test)))

            np.asmatrix( y_pred )
            y_pred = np.asmatrix( y_pred ).transpose()
            y_pred = [Counter(np.asarray(pred).tolist()[0]).most_common(1)[0][0] for pred in y_pred]

            accuracy = accuracy_score(y_test, y_pred)
            results.append(accuracy)

        mean_accuracy = np.mean(results)
        std_dev_accuracy = np.std(results)
        confidence_level = 0.95
        z_value = norm.ppf((1 + confidence_level) / 2)

        margin_of_error = z_value * (std_dev_accuracy / np.sqrt(iterations))
        confidence_interval = (mean_accuracy - margin_of_error, mean_accuracy + margin_of_error)

        print(f'Média da acurácia: {mean_accuracy * 100:.2f}%')
        print(f'Desvio padrão da acurácia: {std_dev_accuracy * 100:.2f}%')
        print(f'Intervalo de Confiança: {confidence_interval}')

        return mean_accuracy, std_dev_accuracy, confidence_interval


class_identifiers = {
    'haberman' : "Survival Status",
    'iris' : 'variety',
    'india_diabetes' : 'class',
    'dermatology' : 'class',
    'ionosphere' : 'class'
}


for (file, identifier) in class_identifiers.items():
    base_name = file
    data_set = f'data sets/{base_name}.csv'
    class_identifier = class_identifiers[base_name]

    classifier = KnnClassifier(data_set = data_set, class_identifier = class_identifier)
    classifier2 = KnnClassifier(data_set = data_set, class_identifier = class_identifier)
    classifier3 = KnnClassifier(data_set = data_set, class_identifier = class_identifier)
    classifier4 = KnnClassifier(data_set = data_set, class_identifier = class_identifier)



    all_results = []

    all_results.append(classifier.validate(method = 'method_1'))
    all_results.append(classifier2.validate(method = 'method_2'))
    all_results.append(classifier3.validate())

    for tc in [2, 3]:

        if tc == 3:
            all_results.append(classifier4.cross_validate())

        method_names = ['Método 1', 'Método 2', 'Sem preparo', 'Classificador\nCombinado']
        means = [result[0] for result in all_results]
        errors = [result[2] for result in all_results]
        colors = ['blue', 'red', 'green', 'purple']

        fig, ax = plt.subplots()


        for i, mean in enumerate(means):
            lower_bound, upper_bound = errors[i]
            ax.scatter(method_names[i], means[i], marker='o', color=colors[i], label=method_names[i])
            ax.fill_between([method_names[i]], lower_bound, upper_bound, color=colors[i], alpha=0.3)

        ax.set_ylabel('Acurácia Média')
        ax.set_title(f'Comparação de Métodos na base "{base_name}"')
        ax.legend()
        plt.savefig(f'images/tc{tc}/{base_name}.pdf', bbox_inches='tight')