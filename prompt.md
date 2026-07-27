我想尽量少对已有的代码进行改动，我现在的想法是，先判断余4是否等于0，如果等于0，那就按照已有的操作来，如果不等于0，那么`allocate_ban_num`就需要在余4等于0的基础上加1。也就是：
```python
if allocate_size_byte % self.access_width_byte == 0:
    allocate_bank_num = allocate_size_byte // self.access_width_byte
else:
    allocate_bank_num = allocate_size_byte // self.access_width_byte + 1
```

