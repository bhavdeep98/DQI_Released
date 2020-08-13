#!/usr/bin/env python
# coding: utf-8

# In[10]:


import time
import multiprocessing
import concurrent.futures


# In[11]:


start = time.perf_counter()


# In[12]:


def sleeping(number,r):
    print('Sleeping for 1 second')
    time.sleep(number)
    return "Done sleeping"


# In[13]:


p1 = multiprocessing.Process(target=sleeping,args=(4,2))
p2 = multiprocessing.Process(target=sleeping,args=(4,2))


# In[14]:


p1.start()
p2.start()

p1.join()
p2.join()


# In[15]:


# with concurrent.futures.ProcessPoolExecutor() as executor:
# #     f1 = executor.submit(sleeping)
# #     f2 = executor.submit(sleeping)
# #     print(f1.result)
# #     print(f2.result)
#     results = [executor.submit(sleeping,1) for _ in range(10)]
    
#     for f in concurrent.futures.as_completed(results):
#         print(f.result())


# In[16]:


# with concurrent.futures.ProcessPoolExecutor() as executor:
#     secs = [5,4,3,2,1]
#     results = executor.map(sleeping,secs)
    
#     for result in results:
#         print(result)


# In[17]:


finish  = time.perf_counter()


# In[18]:


print(f'Finished in {round(finish-start,2)} seconds')


# In[ ]:




