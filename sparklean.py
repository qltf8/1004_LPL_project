import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id
import pyspark
import os
import shutil
from pyspark.sql import SQLContext
from pyspark.sql.functions import *


class cleaner():
    def __init__(self,data,save_dir):
        '''
        Constructor

        Args:
            data: string, path of the data file
            save_dir: string, path for saving different versions of data

        '''
        self.sc = pyspark.SparkContext()
        self.ss = SQLContext(self.sc)
        try:
            self.data=self.ss.read.csv(data)
        except Exception as e:
            self.sc.stop()
            print(e)
            raise ValueError("Can't read file, sparkcontext stopped.")
        self.out='Spark-lean default output'
        if not os.path.isdir(save_dir):
            raise ValueError('Directory does not exist! Create it first!')
        self.save_dir=save_dir

    def initialize(self):
        self.data.printSchema()
        return
    def stop_context(self):
        '''
        Stop the current spark context.
        Call it after done with the class to avoid confliction of multiple spark contexts.

        '''
        self.sc.stop()
        return

    def save_data_version(self,name):

        '''
        Save current data to a checkpoint version. In case you want a backup copy.

        Args:
            name:string,name of this data version.

        '''
        self.data.write.parquet(os.path.join(self.save_dir,name))
        return
    def load_data_version(self, name):

        '''
        Load a saved checkpoint version.

        Args:
            name:string,name of this data version.

        '''
        if not os.path.isdir(os.path.join(self.save_dir,name)):
            raise ValueError('Directory does not exist!')
        self.data=self.ss.read.parquet(os.path.join(self.save_dir,name))
        return
    def delete_data_version(self,name):
        '''
        Delete a saved checkpoint version.

        Args:
            name:string,name of this data version.

        '''
        if not os.path.isdir(os.path.join(self.save_dir,name)):
            raise ValueError('Directory does not exist!')
        shutil.rmtree(os.path.join(self.save_dir,name))
        return
    def detect_missing_value(self,columns='DEFAULT',keywords=["na","","n/a","N/A"]):
        '''
        Provide a missing value diagnosis and further operations.

        Args:
            keywords: list of strings, possible keywords that represent null values.
            columns: list of strings, range of columns to inspect. Default: all columns.

        '''
        output=[]
        targets={}
        null_targets=[]
        if columns=='DEFAULT':
            columns=self.data.columns

        for col in columns:
            n=self.data.where(self.data[col].isNull()).count()
            if n >0:
                message='column '+col+' has '+str(n)+' null values!'
                output.append(message)
                null_targets.append(col)
        for naword in keywords:
            for col in self.data.columns:
                n=self.data.filter(self.data[col]==naword).count()
                if  n> 0:
                    message='Column '+col+' has '+str(n)+' rows that include keyword: '+naword
                    output.append(message)
                    if col in targets:
                        targets[col].append(naword)
                    else:
                        targets[col]=[naword]
        self.out=(null_targets,targets)
        for i in output:
            print(i)
        for i in range(5):
            print('\n')
        print('-'*16)

        print("Please select an approach:")
        print("1. Delete all suspicious rows")
        print("2. Replacing suspicious rows with 0")
        print("3. Replacing suspicious rows with input")
        print("4. Only delete rows with null values")
        print("5. Replacing null values with input")
        print("6. Replacing null values with 0")
        print("7. Do nothing")

        try:
            mode=int(input('Input(number):'))
        except ValueError:
            print ("Not a number")
        # if mode==10:
        #     for col,naword in targets.items():
        #         c.replace(col,naword)
        if mode==1:
            for col in null_targets:
                self.data=self.data.where(self.data[col].isNotNull())
            for k,v in targets.items():
                for value in v:
                    self.remove_by_column_and_value(k,v)
            print("All deleted!")
        elif mode==2:
            self.data=self.data.na.fill('0')
            for k,v in targets.items():
                for value in v:
                    self.remove_by_column_and_value(k,v)
        elif mode==3:
            target=str(input('Input(to replace null):'))
            self.data=self.data.na.fill(target)
            for k,v in targets.items():
                for value in v:
                    self.replace_by_value(k, v, target)
        elif mode==4:
            for col in null_targets:
                self.data=self.data.where(self.data[col].isNotNull())

        elif mode==5:
            target=str(input('Input(to replace null):'))
            self.data=self.data.na.fill(target)

        elif mode==6:
            self.data=self.data.na.fill('0')
        return


    def reset_index(self, drop = False):
        '''
        Reset index of the input

        Args:
            df: input DataFrame object
            drop: whether drop the original index column, default False

        '''
        if drop:
            data = self.data.drop('index')
        else:
            df = self.data.drop('index')
            data = df.select("*").withColumn("index", monotonically_increasing_id())

        self.data = data
        return

    def replace_by_value(self, col_name, org_value, new_value):
        '''
        Replace value of a instance by its column name and original value

        Args:
            df: input DataFrame object
            col_name: column name of the target instance (string)
            org_value: original value of the target instance
            new_value: replaced value
        '''
        if isinstance(col_name, str):
            if col_name in self.data.columns:
                org_value_type = type(org_value)
                if isinstance(self.data.first()[col_name], org_value_type):
                    new_value_type = type(new_value)
                    if org_value_type == new_value_type:
                        data = self.data.replace(org_value, new_value, col_name)
                    else:
                        raise TypeError("{} is {} object, not the same type as {}: {}".format(new_value, new_value_type,org_value,org_value_type))
                else:
                    raise TypeError("{} is {} object, not the same type as {}: {}".format(org_value, type(org_value), col_name, type(self.data.first()[col_name])))
            else:
                raise NameError('{} is not a valid column name'.format(col_name))
        else:
            raise TypeError('column name must be a string object. your input type is {}'.format(type(col_name)))

        self.data = data
        return

    def replace_by_index(self, col_name, index, new_value):
        '''
        Replace value of a instance by its column name and index

        Args:
            df: input DataFrame object
            col_name: column name of the target instance (string)
            index: index of the target instance(int)
            new_value: replaced value

        '''
        if isinstance(col_name, str):
            if "index" in self.data.columns:
                if isinstance(index, int):
                    new_value_type = type(new_value)
                    if isinstance(self.data.first()[col_name], new_value_type):
                        data = self.data.withColumn(col_name,when(self.data["index"] == index, new_value).otherwise(self.data[col_name]))
                    else:
                        raise TypeError("{} is {} object, not the same type as {}: {}".format(new_value, type(new_value), col_name, type(self.data.first()[col_name])))
                else:
                    raise TypeError("index must be a int object. your input type is {}".format(type(index)))
            else:
                raise ValueError('The input DataFrame object does not contain a index column')
        else:
            raise TypeError('column name must be a string object. your input type is {}'.format(type(col_name)))

        self.data = data
        return
    def remove_by_index(self,index):
        df = self.data
        if 'index' in df.columns:
            pass
        else:
            return "DataFrame dosen't have index"

        df.createOrReplaceTempView("df")
        num_before_remove = df.count()
        if isinstance(index,tuple):
            pass
        elif isinstance(index,list):
            if len(index) == 0:
                return "Empty list"
            elif len(index) == 1:
                index = index[0]
                df = self.ss.sql("select * from df where index != '{}'".format(index))
                num_after_remove = df.count()
                self.data = df
                print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
                return
            else:
                index = tuple(index)
        elif isinstance(index, int) or isinstance(index,str):
            df = self.ss.sql("select * from df where index != '{}'".format(index))
            num_after_remove = df.count()
            self.data = df
            print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
            return
        else:
            return "Wrong index type, only list, tuple, str, or int is accepted, you input type is {}".format(type(index))

        df = self.ss.sql("select * from df where index not in {}".format(index))
        num_after_remove = df.count()
        self.data = df
        print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
        return



    def remove_by_column_and_value(self,column,values):
        '''
        df : dataframe
        column: str
        values: number/string
        '''
        df = self.data
        if isinstance(column,str):
            pass
        else:
            return 'column must be str type'
        if column in df.columns:
            pass
        else:
            return 'feature not in scheme'
        if isinstance(values, int) or isinstance(values,str):
            pass
        else:
            return 'The value mush be str or number, your input is'+str(type(values))
        df.createOrReplaceTempView("df")
        num_before_remove = df.count()
        the_number_of_same_value = self.ss.sql("select * from df where  {} = '{}'".format(column,values)).count()
        if the_number_of_same_value > 0:
            df = self.ss.sql("select * from df where  {} != '{}'".format(column,values))
        num_after_remove = df.count()
        self.data = df
        print('The number of rows removed : {}'.format(num_before_remove-num_after_remove))
        return
    def dropDuplicateColumn(self):
        df = self.data
        i = 0
        df.createOrReplaceTempView("df")
        total_row = df.count()
        while i < len(df.columns)-1:
            df_col_len = len(df.columns)
            left_col = df.columns[i]
            print("Checking column {}".format(left_col))
            left_table = self.ss.sql("select {} from df".format(left_col))
            left_table.createOrReplaceTempView("left_table")
            j = i + 1
            while j < len(df.columns):
                right_col = df.columns[j]
                right_table = self.ss.sql("select {} from df".format(right_col))
                right_table.createOrReplaceTempView("right_table")
                #firt, find if the first num_try of row is same
                n_try = 10
                left_temp_n_try = left_table.take(n_try)
                right_temp_n_try = right_table.take(n_try)
                indicator = False
                for l_row,r_row in zip(left_temp_n_try,right_temp_n_try):
                    if (l_row[left_col]) == (r_row[right_col]):
                        indicator = True
                        pass
                    else:
                        indicator = False
                        #print(left_col,right_col,'not equal')
                        break
                #left_table and right_table have same first 10 value
                if indicator:
                    the_number_of_same_line = self.ss.sql("select * from left_table inner join right_table on {}={}".format(left_col,right_col)).count()
                    #print(the_number_of_same_line,total_row)
                    if the_number_of_same_line == total_row:
                        option = input("Column {} and column {} are same.\n \
                        Press 1 to drop {}, press 2 to drop {}, press 3 or other to keep both".format(left_col,right_col,left_col,right_col))
                        if option =='1':
                            print('The column',left_col,'will be dropped')
                            currenct_list_columns = df.columns
                            currenct_list_columns.remove(left_col)
                            str_currenct_list_columns = ','.join(currenct_list_columns)
                            df = self.ss.sql("select {} from df".format(str_currenct_list_columns))
                            df.createOrReplaceTempView("df")
                            i = i - 1
                            print('Success')
                            break
                        elif option == '2':
                            print('The column',right_col,'will be dropped')
                            currenct_list_columns = df.columns
                            currenct_list_columns.remove(right_col)
                            str_currenct_list_columns = ','.join(currenct_list_columns)
                            df = self.ss.sql("select {} from df".format(str_currenct_list_columns))
                            df.createOrReplaceTempView("df")
                            j = j - 1
                            print('Success')
                        else:
                            print("Not column will be dropped")
                    else:
                        #print('10 same, not same')
                        pass
                else:
                    #print('10 not same, not same')
                    pass
                j = j + 1
            i = i + 1
        self.data =df
        return
    def dropColumnWithAllTheSameValues(self):
        df = self.data
        df.createOrReplaceTempView("df")
        for col in df.columns:
            print("Checking column {}".format(col))
            first_set = set()
            num_of_row = df.count()
            num_first_try = 10
            temp_df = self.ss.sql("select {} from df limit 10".format(col)).take(num_first_try)
            for i in range(num_first_try):
                first_set.add(temp_df[i][col])
            if len(first_set) == 1:
                target_value = first_set.pop()
                num_of_row_with_same_value = self.ss.sql("select * from df where {} = '{}'".format(col,target_value)).count()
                if num_of_row_with_same_value == df.count():
                    option = input("Colume {} has the same value for all cells, do you want to drop it?\n\
                    Press 1 to drop it, press 2 or other to keep it")
                    if option == '1':
                        print('The column',col,'will be dropped')
                        currenct_list_columns = df.columns
                        currenct_list_columns.remove(col)
                        str_currenct_list_columns = ','.join(currenct_list_columns)
                        df = self.ss.sql("select {} from df".format(str_currenct_list_columns))
                        df.createOrReplaceTempView("df")
                    else:
                        pass
            else:
                pass
                #print('not same')
        self.data =df
        return
