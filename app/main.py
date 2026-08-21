from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="FastAPI Todo App")

class Todo(BaseModel):
    id: Optional[int] = None
    title: str
    completed: bool = False

todos: List[Todo] = []

@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI Todo App"}

@app.get("/todos", response_model=List[Todo])
def get_todos():
    return todos

@app.post("/todos", response_model=Todo)
def create_todo(todo: Todo):
    todo.id = len(todos) + 1
    todos.append(todo)
    return todo

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    for item in todos:
        if item.id == todo_id:
            return item
    raise HTTPException(status_code=404, detail="Todo not found")

@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, updated: Todo):
    for idx, item in enumerate(todos):
        if item.id == todo_id:
            updated.id = todo_id
            todos[idx] = updated
            return updated
    raise HTTPException(status_code=404, detail="Todo not found")

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    for idx, item in enumerate(todos):
        if item.id == todo_id:
            todos.pop(idx)
            return {"message": f"Todo {todo_id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Todo not found")
