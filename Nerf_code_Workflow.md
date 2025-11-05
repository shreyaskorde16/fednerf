Nerf Workflow:





def Train()

&nbsp;	load data()

&nbsp;	cast intrinsics (H;W;FOCAL) in Right types

&nbsp;	

178	**create nerf model(cfg)** -> render\_kwargs\_train, kwargs\_test, start, grad\_vars\_ Optimizer



&nbsp;	if render True: **render\_only()** -> renders novel views for val data



&nbsp;	Random ray batching

&nbsp;	

	**for loop for Iteration (Training):**



&nbsp;		# core optimization loop

69		rgb, disp, acc, extras = **render**(H, W, K, chunk=args.chunk, rays=batch\_rays,

&nbsp;                                               verbose=i < 10, retraw=True,

&nbsp;                                               \*\*render\_kwargs\_train)



&nbsp;			# aall ret contains \['rgb\_map', 'disp\_map', 'acc\_map']

126			(Output) all\_ret = **batchify\_rays**(rays, chunk, \*\*kwargs)



&nbsp;				# Render rays in smaller minibatches to avoid OOM.

59				ret = **render\_rays**(rays\_flat\[i:i+chunk], \*\*kwargs)





308					#def **render\_rays(**ray\_batch,

&nbsp;                                                        network\_fn,

&nbsp;							 network\_query\_fn,

&nbsp;                                                        N\_samples,

&nbsp;                                                        retraw=False,

&nbsp;          					         lindisp=False,

&nbsp;          						 perturb=0.,

&nbsp;          					         N\_importance=0,

&nbsp;           					         network\_fine=None,

&nbsp;             						 white\_bkgd=False,

&nbsp;             					         raw\_noise\_std=0.,

&nbsp;             					         verbose=False,

&nbsp;             						 pytest=False): -> 

&nbsp;					**Returns:**

&nbsp;    					rgb\_map: \[num\_rays, 3]. Estimated RGB color of a ray. Comes from fine             					model.

&nbsp;   					disp\_map: \[num\_rays]. Disparity map. 1 / depth.

&nbsp;     					acc\_map: \[num\_rays]. Accumulated opacity along each ray. Comes from fine 					model.

&nbsp;     					raw: \[num\_rays, num\_samples, 4]. Raw predictions from model.

&nbsp;     					rgb0: See rgb\_map. Output for coarse model.

&nbsp;     					disp0: See disp\_map. Output for coarse model.

&nbsp;     					acc0: See acc\_map. Output for coarse model.

&nbsp;     					z\_std: \[num\_rays]. Standard deviation of distances along ray for each

&nbsp;      					sample.

&nbsp;					

**385\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_raw** = **network\_query\_fn**(pts, viewdirs, network\_fn) -> **run\_network (lambda):**



&nbsp;						# run network **(coarse Nerf model)**

37						def **run\_network**(inputs, viewdirs, fn, embed\_fn, embeddirs\_fn, 

&nbsp;								netchunk=1024\*64): 

&nbsp;						|

&nbsp;						|	**outputs\_flat = batchify(**fn, netchunk)(embedded)

&nbsp;						|

&nbsp;						|       def **batchify(fn, chunk):** 

&nbsp;						|	  def **ret(inputs)**:

&nbsp;						|	  torch.cat(\[fn(inputs\[i:i+chunk]) for i in range(0, 

&nbsp;                                               |         |     inputs.shape\[0], chunk)], 0) # model output flat

&nbsp;						|	  return -> **ret function**

						**return -> Outputs**



**386\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**\_rgb\_map, disp\_map, acc\_map, weights, depth\_map **= raw2outputs**(raw, 

&nbsp;                                                                                z\_vals, rays\_d, raw\_noise\_std,

&nbsp;                                                                                white\_bkgd, pytest=pytest)



						# Transforms model's predictions to semantically meaningful values.

**262**						def **raw2outputs**(raw, z\_vals, rays\_d, raw\_noise\_std=0, 

&nbsp;                                                                 white\_bkgd=False, pytest=False):



**126\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ all\_ret as Output** 

**760\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ rgb, disp, acc, extras from 134 line**





















&nbsp;		# logging

&nbsp;		rgbs, disps = render\_path(render\_poses, hwf, K, args.chunk, render\_kwargs\_test)

&nbsp;	

&nbsp;	



~~~~~~~~~~~~~~



inside create nerf:



network\_query\_fn = lambda inputs, viewdirs, network\_fn : run\_network(inputs, viewdirs, network\_fn,

&nbsp;                                                               embed\_fn=embed\_fn,

&nbsp;                                                               embeddirs\_fn=embeddirs\_fn,

&nbsp;                                                               netchunk=args.netchunk)

&nbsp;		

&nbsp;render\_kwargs\_train = {

&nbsp;       'network\_query\_fn' : network\_query\_fn,

&nbsp;       'perturb' : args.perturb,

&nbsp;       'N\_importance' : args.N\_importance,

&nbsp;       'network\_fine' : model\_fine,

&nbsp;       'N\_samples' : args.N\_samples,

&nbsp;       'network\_fn' : model,

&nbsp;       'use\_viewdirs' : args.use\_viewdirs,

&nbsp;       'white\_bkgd' : args.white\_bkgd,

&nbsp;       'raw\_noise\_std' : args.raw\_noise\_std,

&nbsp;   }

























































