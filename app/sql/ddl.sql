drop database abandoned_object_detect;
create database abandoned_object_detect;
use abandoned_object_detect;
CREATE TABLE user
(
    id       INT PRIMARY KEY AUTO_INCREMENT, -- 用户id
    username VARCHAR(20),                    -- 登录名称
    password VARCHAR(20),                     -- 密码
    insert_time datetime not null,
    update_time datetime not null
); -- 用户表


INSERT INTO user(username,PASSWORD,insert_time,update_time) VALUES('root','root',NOW(),NOW());

create table video_resource
(
    id       INT PRIMARY KEY AUTO_INCREMENT,
    file_name varchar(20),
    path varchar(50),
    insert_time datetime not null,
    update_time datetime not null
)