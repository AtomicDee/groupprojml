all_data = pd.read_csv('drawem_labels_all_data.csv')
print 'all data : ', GA_all_data
Id = datainfo.iloc[0:len(datainfo), 0]
G = datainfo.iloc[0:len(datainfo), 1]
GAbirth = datainfo.iloc[0:len(datainfo), 2]
GAscan = datainfo.iloc[0:len(datainfo), 3]
region = datainfo.iloc[0:len(datainfo), 4]
t1 = datainfo.iloc[0:len(datainfo), 5]
t2 = datainfo.iloc[0:len(datainfo), 6]
v = datainfo.iloc[0:len(datainfo), 7]
reduced_data_g.append(G,region,t1)
reduced_data_b.append(GAbirth,region,t1)
dg = []
titles_g = [Gender','Region', 'T1 Average Intensity']
titles_b = ['Birth Age','Region', 'T1 Average Intensity']
db = []
dg.append(pd.DataFrame(reduced_data_g, columns = titles_g))
db.append(pd.DataFrame(reduced_data_g, columns = titles_b))


results_g = pd.concat(dg,titles_g)
results_b = pd.concat(db,titles_b)

results_g.to_csv('gender_data.csv')
results_b.to_csv('birth_data.csv')
