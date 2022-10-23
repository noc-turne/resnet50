srun -p platform_test --gres=gpu:8 -n3 --ntasks-per-node=2 python torch_doc.py
#srun -p platform_test --gres=gpu:8 -n2 --ntasks-per-node=1 python pape_doc.py
