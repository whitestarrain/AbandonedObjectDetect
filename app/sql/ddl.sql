create database abandoned_object_detect;
use abandoned_object_detect;
CREATE TABLE USER
(
    id       INT PRIMARY KEY AUTO_INCREMENT, -- 用户id
    username VARCHAR(20),                    -- 登录名称
    password  VARCHAR(20)                     -- 密码
); -- 用户表
